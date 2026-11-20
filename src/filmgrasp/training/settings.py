"""Training CLI parsing and persisted experiment settings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from filmgrasp.training.losses import validate_loss_settings
from filmgrasp.utils import file_sha256

CONFIG_PATH_FIELDS = {"train_root", "val_root", "output_root", "pretrained", "resume"}


def dropout_settings(config):
    """Resolve saved settings; checkpoints predating A1 used Dropout OFF."""
    enabled = config.get("dropout_enabled", False)
    probability = config.get("dropout_prob", 0.0)
    valid_probability = (
        isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and 0 <= probability < 1
    )
    if not isinstance(enabled, bool) or not valid_probability:
        raise ValueError("Invalid Dropout configuration")
    if enabled != (probability > 0):
        raise ValueError("Inconsistent Dropout configuration")
    return enabled, float(probability)


def loss_settings(config):
    """Resolve settings for legacy B0/A1 checkpoints with no loss metadata."""
    return validate_loss_settings(
        config.get("loss_type", "b0"),
        config.get("lambda_bg", 0.05),
    )


def backbone_init_mode(config):
    """Resolve initialization for legacy checkpoints, which used pretrained weights."""
    mode = config.get("backbone_init", "pretrained")
    if mode not in ("pretrained", "random"):
        raise ValueError("Invalid backbone initialization mode: {}".format(mode))
    return mode


def _config_defaults(path, parser):
    """Load checked CLI defaults from a JSON experiment configuration."""
    path = path.expanduser().resolve()
    if not path.is_file():
        parser.error("Configuration file not found: {}".format(path))
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error("Cannot read configuration {}: {}".format(path, exc))
    if not isinstance(values, dict):
        parser.error("Configuration must be a JSON object: {}".format(path))

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in ("help", "config")
    }
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error("Unknown configuration keys: {}".format(", ".join(unknown)))

    checked = {}
    for key, value in values.items():
        action = actions[key]
        if key in CONFIG_PATH_FIELDS:
            if value is None and key in ("pretrained", "resume"):
                checked[key] = None
                continue
            if not isinstance(value, str) or not value:
                parser.error(
                    "Configuration key '{}' must be a non-empty path string".format(key)
                )
            value = Path(value).expanduser()
            checked[key] = value if value.is_absolute() else (path.parent / value).resolve()
            continue
        if action.type is int and (not isinstance(value, int) or isinstance(value, bool)):
            parser.error("Configuration key '{}' must be an integer".format(key))
        if action.type is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                parser.error("Configuration key '{}' must be a number".format(key))
            value = float(value)
        if action.type is None and not isinstance(value, str):
            parser.error("Configuration key '{}' must be a string".format(key))
        if action.choices is not None and value not in action.choices:
            parser.error(
                "Invalid value for '{}': {!r}; choose from {}".format(
                    key, value, ", ".join(action.choices)
                )
            )
        checked[key] = value
    return path, checked


def _build_parser():
    parser = argparse.ArgumentParser(description="Train FiLM-GraspNet.")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON experiment config; explicit CLI options override its values",
    )
    parser.add_argument("--train-root", type=Path, default=Path("../storage/data/train_debug"))
    parser.add_argument("--val-root", type=Path, default=Path("../storage/data/val_debug"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="filmgrasp_debug01")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="GRConvNet3 weights; omit to train the backbone from random initialization",
    )
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=("b0", "masked_geometry"), default="b0")
    parser.add_argument("--lambda-bg", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path)
    probed, _ = config_probe.parse_known_args(argv)
    if probed.config is None:
        config_path = None
    else:
        config_path, defaults = _config_defaults(probed.config, parser)
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args.config = config_path
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; num-workers must be nonnegative")
    if not 0 <= args.dropout_prob < 1:
        parser.error("dropout-prob must satisfy 0 <= p < 1")
    try:
        validate_loss_settings(args.loss_type, args.lambda_bg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def build_run_config(args, device):
    """Build the complete, environment-resolved metadata saved with a run."""
    dropout_enabled = args.dropout_prob > 0
    config = {
        "model": "FiLMGraspNet",
        "backbone": "GRConvNet3 RGB 32 channels dropout={}".format(dropout_enabled),
        "backbone_init": "pretrained" if args.pretrained is not None else "random",
        "dropout_enabled": dropout_enabled,
        "dropout_prob": args.dropout_prob,
        "loss_type": args.loss_type,
        "lambda_bg": args.lambda_bg,
        "clip_model": "ViT-B/32",
        "clip_device": "cpu",
        "input_resolution": [416, 416],
        "optimizer": "Adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "train_root": str(args.train_root),
        "val_root": str(args.val_root),
        "pretrained": str(args.pretrained) if args.pretrained is not None else None,
        "pretrained_sha256": file_sha256(args.pretrained) if args.pretrained is not None else None,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    if args.config is not None:
        config["source_config"] = str(args.config)
        config["source_config_sha256"] = file_sha256(args.config)
    return config
