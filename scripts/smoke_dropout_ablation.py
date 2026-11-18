"""Small A1 configuration and checkpoint smoke tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.training.train import (
    _checkpoint, build_model, dropout_settings, load_checkpoint, parse_args,
)


PRETRAINED = Path(__file__).resolve().parents[1] / "checkpoints/grconvnet3_pretrained.pth"


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.embedding_dim = 512
        self.clip_model = torch.nn.Linear(1, 1).to(device)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, prompts):
        return torch.ones(len(prompts), self.embedding_dim)


class DropoutAblationTests(unittest.TestCase):
    def test_cli_and_legacy_defaults(self):
        self.assertEqual(parse_args(["--device", "cpu"]).dropout_prob, 0.0)
        self.assertEqual(dropout_settings({}), (False, 0.0))
        for value in ("-0.1", "1", "nan"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args(["--device", "cpu", "--dropout-prob", value])

    def test_model_and_modes(self):
        torch.manual_seed(42)
        with patch("filmgrasp.training.train.CLIPTextEncoder", FakeTextEncoder):
            b0 = build_model(PRETRAINED, "cpu")
            a1 = build_model(PRETRAINED, "cpu", True, 0.1)
        self.assertFalse(b0.backbone.dropout)
        self.assertTrue(a1.backbone.dropout)
        modules = [module for module in a1.backbone.modules()
                   if isinstance(module, torch.nn.Dropout)]
        self.assertEqual(len(modules), 4)
        self.assertTrue(all(module.p == 0.1 for module in modules))
        self.assertEqual(set(b0.backbone.state_dict()), set(a1.backbone.state_dict()))
        self.assertTrue(all(torch.equal(b0.backbone.state_dict()[key], value)
                            for key, value in a1.backbone.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for p in a1.text_encoder.parameters()))
        self.assertTrue(all(p.device.type == "cpu" for p in a1.text_encoder.parameters()))

        image = torch.randn(1, 3, 64, 64)
        a1.train()
        self.assertFalse(a1.text_encoder.training)
        with torch.no_grad():
            train_first = a1(image, ["grasp"])
            train_second = a1(image, ["grasp"])
        self.assertEqual([x.shape for x in train_first], [x.shape for x in train_second])
        self.assertEqual([x.shape for x in train_first], [(1, 1, 64, 64)] * 4)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(train_first, train_second)))
        a1.eval()
        with torch.no_grad():
            eval_first = a1(image, ["grasp"])
            eval_second = a1(image, ["grasp"])
            baseline = b0.eval()(image, ["grasp"])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(eval_first, eval_second)))
        self.assertEqual([x.shape for x in baseline], [x.shape for x in eval_first])

    def test_checkpoint_metadata_and_resume_guards(self):
        config = {"pretrained_sha256": "same", "batch_size": 4, "lr": 1e-4,
                  "seed": 42, "optimizer": "Adam", "clip_model": "ViT-B/32",
                  "epochs": 2, "dropout_enabled": True, "dropout_prob": 0.1}
        legacy = {key: value for key, value in config.items()
                  if not key.startswith("dropout_")}
        summary = {name: {"manifest_sha256": name} for name in ("train", "validation")}
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            a1 = _checkpoint(model, optimizer, 1, 0.0, 1.0, config, summary)
            self.assertEqual((a1["dropout_enabled"], a1["dropout_prob"]), (True, 0.1))
            self.assertEqual(dropout_settings(a1["config"]), (True, 0.1))
            torch.save(a1, path)
            with self.assertRaisesRegex(ValueError, "Dropout"):
                load_checkpoint(path, model, optimizer, legacy, summary)
            b0 = _checkpoint(model, optimizer, 1, 0.0, 1.0, legacy, summary)
            b0.pop("dropout_enabled")
            b0.pop("dropout_prob")
            torch.save(b0, path)
            with self.assertRaisesRegex(ValueError, "Dropout"):
                load_checkpoint(path, model, optimizer, config, summary)
            self.assertEqual(load_checkpoint(path, model, optimizer, legacy, summary)["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
