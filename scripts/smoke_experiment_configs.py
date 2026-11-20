"""Validate versioned experiment configs and config/CLI precedence."""

import json
import tempfile
import unittest
from pathlib import Path

from filmgrasp.training.train import parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class ExperimentConfigTests(unittest.TestCase):
    def test_pilot_configs(self):
        expected = {
            "pilot_b0.json": (0.0, "b0"),
            "pilot_a1_dropout.json": (0.1, "b0"),
            "pilot_a2_masked_geometry.json": (0.1, "masked_geometry"),
        }
        for name, settings in expected.items():
            with self.subTest(name=name):
                args = parse_args([
                    "--config", str(CONFIGS / name),
                    "--device", "cpu",
                ])
                self.assertEqual((args.dropout_prob, args.loss_type), settings)
                self.assertTrue(args.train_root.is_dir())
                self.assertTrue(args.val_root.is_dir())
                self.assertTrue(args.pretrained.is_file())
                self.assertEqual(args.device, "cpu")

    def test_relative_paths_and_cli_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "experiment.json"
            config_path.write_text(json.dumps({
                "train_root": "../data/train",
                "val_root": "../data/val",
                "output_root": "../outputs",
                "epochs": 100,
                "batch_size": 4,
                "device": "cpu",
                "pretrained": None,
                "run_name": "from_config",
            }), encoding="utf-8")

            args = parse_args(["--config", str(config_path), "--epochs", "3"])

            self.assertEqual(args.config, config_path.resolve())
            self.assertEqual(args.train_root, (root / "data/train").resolve())
            self.assertEqual(args.val_root, (root / "data/val").resolve())
            self.assertEqual(args.output_root, (root / "outputs").resolve())
            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.run_name, "from_config")

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "invalid.json"
            config_path.write_text(json.dumps({"epochz": 100}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()
