"""Lightweight checkpoint compatibility and restoration checks."""

import csv
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from filmgrasp.training.train import (
    COLUMNS,
    _checkpoint,
    load_checkpoint,
    load_trusted_training_checkpoint,
    main,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.film = torch.nn.Linear(2, 1)
        self.text_encoder = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.film(self.backbone(x))


def make_optimizer(model):
    return torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.film.parameters()), lr=1e-4
    )


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(self.restore_original_rng)
        self.path = Path(self.folder.name) / "last.pt"
        self.config = {
            "pretrained_sha256": "trusted-backbone-hash",
            "batch_size": 1,
            "lr": 1e-4,
            "seed": 42,
            "optimizer": "Adam",
            "clip_model": "ViT-B/32",
            "epochs": 2,
            "torch_version": torch.__version__,
        }
        self.data_summary = {
            "train": {"manifest_sha256": "train-hash"},
            "validation": {"manifest_sha256": "validation-hash"},
        }

    def restore_original_rng(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)

    def make_checkpoint(self):
        random.seed(789)
        np.random.seed(456)
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(321)
        model = TinyModel()
        optimizer = make_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved = _checkpoint(model, optimizer, 1, 12.5, 0.25,
                            self.config, self.data_summary)
        torch.save(saved, self.path)
        return model, optimizer, saved

    def test_trusted_loader_uses_supported_api_without_default_behavior(self):
        self.make_checkpoint()
        real_load = torch.load
        calls = []

        def modern_load(path, *, map_location, weights_only):
            calls.append((map_location, weights_only))
            return real_load(path, map_location=map_location)

        with patch("filmgrasp.training.train.torch.load", modern_load):
            saved = load_trusted_training_checkpoint(self.path)
        self.assertEqual(calls, [("cpu", False)])
        self.assertEqual(saved["epoch"], 1)

        def old_load(path, *, map_location):
            calls.append((map_location, "absent"))
            return real_load(path, map_location=map_location)

        with patch("filmgrasp.training.train.torch.load", old_load):
            load_trusted_training_checkpoint(self.path)
        self.assertEqual(calls[-1], ("cpu", "absent"))

    def test_restores_model_optimizer_metrics_epoch_and_rng(self):
        original, _, saved = self.make_checkpoint()
        random.random()
        np.random.rand()
        torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1, device="cuda")

        restored = TinyModel()
        optimizer = make_optimizer(restored)
        result = load_checkpoint(self.path, restored, optimizer,
                                 self.config, self.data_summary)

        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["epoch"] + 1, 2)
        self.assertEqual(result["best_val_accuracy"], 12.5)
        self.assertEqual(result["best_val_loss"], 0.25)
        for name, parameter in original.state_dict().items():
            if not name.startswith("text_encoder."):
                self.assertTrue(torch.equal(restored.state_dict()[name], parameter))
        self.assertTrue(optimizer.state)
        for original_state, restored_state in zip(saved["optimizer_state_dict"]["state"].values(),
                                                  optimizer.state.values()):
            self.assertEqual(original_state.keys(), restored_state.keys())
            for key in original_state:
                self.assertTrue(torch.equal(original_state[key], restored_state[key]))
        self.assertEqual(random.getstate(), saved["python_rng_state"])
        actual_numpy = np.random.get_state()
        expected_numpy = saved["numpy_rng_state"]
        self.assertEqual(actual_numpy[0], expected_numpy[0])
        np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
        self.assertEqual(actual_numpy[2:], expected_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), saved["rng_state"]))
        if torch.cuda.is_available():
            for actual, expected in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng_state"]):
                self.assertTrue(torch.equal(actual, expected))
        optimizer.zero_grad(set_to_none=True)
        restored(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    def test_existing_compatibility_guards(self):
        model, _, saved = self.make_checkpoint()
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.path.with_name("missing.pt"), model, make_optimizer(model),
                            self.config, self.data_summary)

        cases = [
            ({**self.config, "pretrained_sha256": "different"}, self.data_summary, ValueError),
            ({**self.config, "batch_size": 2}, self.data_summary, ValueError),
            ({**self.config, "epochs": 1}, self.data_summary, ValueError),
            (self.config, {"train": {"manifest_sha256": "different"},
                           "validation": self.data_summary["validation"]}, ValueError),
        ]
        for config, summary, error in cases:
            with self.subTest(config=config, summary=summary):
                candidate = TinyModel()
                with self.assertRaises(error):
                    load_checkpoint(self.path, candidate, make_optimizer(candidate), config, summary)

        torch.save({"epoch": 1}, self.path)
        with self.assertRaises((KeyError, ValueError, AttributeError)):
            load_checkpoint(self.path, model, make_optimizer(model),
                            self.config, self.data_summary)

    def test_main_resumes_epoch_one_as_epoch_two_without_duplicate_csv_row(self):
        original, _, saved = self.make_checkpoint()
        root = Path(self.folder.name)
        pretrained = root / "backbone.pth"
        pretrained.write_bytes(b"synthetic backbone identifier")
        saved["config"]["pretrained_sha256"] = hashlib.sha256(pretrained.read_bytes()).hexdigest()
        output_root = root / "runs"
        run_dir = output_root / "synthetic_resume"
        checkpoint_path = run_dir / "checkpoints" / "last.pt"
        checkpoint_path.parent.mkdir(parents=True)
        torch.save(saved, checkpoint_path)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow({"epoch": 1})

        train_loader = SimpleNamespace(sampler=SimpleNamespace(generator=torch.Generator()))
        observed = []

        def train_epoch(model, loader, optimizer, device):
            self.assertIs(loader, train_loader)
            self.assertTrue(optimizer.state)
            self.assertTrue(torch.equal(model.backbone.weight, original.backbone.weight))
            observed.append("trained")
            return {"total": 0.4, "pos": 0.1, "cos": 0.1, "sin": 0.1, "width": 0.1}, 1

        validation = {"total": 0.3, "pos": 0.1, "cos": 0.1, "sin": 0.05,
                      "width": 0.05, "accuracy": 15.0, "detection_rate": 100.0,
                      "mean_max_iou": 0.2, "evaluated_samples": 1}
        argv = ["--train-root", str(root / "train"), "--val-root", str(root / "val"),
                "--epochs", "2", "--device", "cpu", "--num-workers", "0",
                "--pretrained", str(pretrained), "--output-root", str(output_root),
                "--run-name", "synthetic_resume", "--resume", str(checkpoint_path)]

        with patch("filmgrasp.training.train.create_dataloaders",
                   return_value=(train_loader, object(), self.data_summary)), \
             patch("filmgrasp.training.train.build_model", side_effect=lambda *_: TinyModel()), \
             patch("filmgrasp.training.train.train_one_epoch", side_effect=train_epoch), \
             patch("filmgrasp.training.train.validate_one_epoch", return_value=validation):
            self.assertEqual(main(argv), run_dir)

        self.assertEqual(observed, ["trained"])
        with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
            self.assertEqual([int(row["epoch"]) for row in csv.DictReader(file)], [1, 2])
        self.assertEqual(load_trusted_training_checkpoint(checkpoint_path)["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
