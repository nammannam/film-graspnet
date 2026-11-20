"""Checkpoint serialization and resume compatibility checks."""

import inspect
import random

import numpy as np
import torch

from filmgrasp.training.settings import backbone_init_mode, dropout_settings, loss_settings

RESUME_CONFIG_KEYS = (
    "pretrained_sha256",
    "batch_size",
    "lr",
    "seed",
    "optimizer",
    "clip_model",
)


def _checkpoint(model, optimizer, epoch, best_accuracy, best_loss, config, data_summary):
    dropout_enabled, dropout_prob = dropout_settings(config)
    loss_type, lambda_bg = loss_settings(config)
    model_state = {
        name: value.cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("text_encoder.")
    }
    return {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_accuracy,
        "best_val_loss": best_loss,
        "config": config,
        "data_summary": data_summary,
        "dropout_enabled": dropout_enabled,
        "dropout_prob": dropout_prob,
        "loss_type": loss_type,
        "lambda_bg": lambda_bg,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path, model, optimizer, epoch, best_accuracy, best_loss, config, data_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        _checkpoint(model, optimizer, epoch, best_accuracy, best_loss, config, data_summary),
        temporary,
    )
    temporary.replace(path)


def load_trusted_training_checkpoint(path):
    """Load a full project-generated checkpoint from a trusted source only."""
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # Full checkpoints contain NumPy RNG state; never use this for untrusted files.
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _validate_checkpoint_metadata(saved, config, data_summary):
    old_config = saved.get("config", {})
    old_dropout = dropout_settings(old_config)
    old_loss = loss_settings(old_config)

    if (saved.get("loss_type", old_loss[0]), saved.get("lambda_bg", old_loss[1])) != old_loss:
        raise ValueError("Checkpoint loss metadata mismatch")
    if old_loss != loss_settings(config):
        raise ValueError("Resume configuration mismatch: loss settings")
    if (
        saved.get("dropout_enabled", old_dropout[0]),
        saved.get("dropout_prob", old_dropout[1]),
    ) != old_dropout:
        raise ValueError("Checkpoint Dropout metadata mismatch")
    if old_dropout != dropout_settings(config):
        raise ValueError("Resume configuration mismatch: Dropout")
    if backbone_init_mode(old_config) != backbone_init_mode(config):
        raise ValueError("Resume configuration mismatch: backbone initialization")

    for key in RESUME_CONFIG_KEYS:
        if old_config.get(key) != config.get(key):
            raise ValueError("Resume configuration mismatch: {}".format(key))
    for split in ("train", "validation"):
        old_hash = saved["data_summary"][split]["manifest_sha256"]
        if old_hash != data_summary[split]["manifest_sha256"]:
            raise ValueError("Resume dataset mismatch: {}".format(split))
    if saved["epoch"] < 1 or saved["epoch"] >= config["epochs"]:
        raise ValueError(
            "Invalid resume epoch {} for {} epochs".format(saved["epoch"], config["epochs"])
        )


def validate_evaluation_checkpoint(saved, run_config):
    """Validate run/checkpoint settings and return resolved Dropout/loss values."""
    saved_config = saved.get("config", {})
    if backbone_init_mode(saved_config) != backbone_init_mode(run_config):
        raise ValueError("Run and checkpoint backbone initialization mismatch")

    saved_dropout = dropout_settings(saved_config)
    embedded_dropout = (
        saved.get("dropout_enabled", saved_dropout[0]),
        saved.get("dropout_prob", saved_dropout[1]),
    )
    if embedded_dropout != saved_dropout:
        raise ValueError("Checkpoint Dropout metadata mismatch")
    if dropout_settings(run_config) != saved_dropout:
        raise ValueError("Run and checkpoint Dropout configuration mismatch")

    saved_loss = loss_settings(saved_config)
    embedded_loss = (
        saved.get("loss_type", saved_loss[0]),
        saved.get("lambda_bg", saved_loss[1]),
    )
    if embedded_loss != saved_loss:
        raise ValueError("Checkpoint loss metadata mismatch")
    if loss_settings(run_config) != saved_loss:
        raise ValueError("Run and checkpoint loss configuration mismatch")
    return saved_dropout, saved_loss


def load_checkpoint(path, model, optimizer, config, data_summary):
    """Restore a trusted checkpoint after validating run and dataset identity."""
    if not path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(path))
    saved = load_trusted_training_checkpoint(path)
    _validate_checkpoint_metadata(saved, config, data_summary)

    missing, unexpected = model.load_state_dict(saved["model_state_dict"], strict=False)
    invalid_missing = [name for name in missing if not name.startswith("text_encoder.")]
    if unexpected or invalid_missing:
        raise ValueError(
            "Incompatible model checkpoint: missing={} unexpected={}".format(
                invalid_missing, unexpected
            )
        )

    optimizer.load_state_dict(saved["optimizer_state_dict"])
    torch.set_rng_state(saved["rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    random.setstate(saved["python_rng_state"])
    if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    return saved
