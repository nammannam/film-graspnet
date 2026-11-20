"""Train and validate FiLM-GraspNet on manifest-based datasets."""

import csv
import json
import logging
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.factory import build_model as _build_model
from filmgrasp.training.checkpoints import (
    _checkpoint,
    load_checkpoint,
    load_trusted_training_checkpoint,
    save_checkpoint,
)
from filmgrasp.training.engine import (
    A2_DIAGNOSTICS,
    _add_loss,
    _averages,
    _collate,
    _loss_totals,
    _prepare,
    aggregate_grasps,
    create_dataloaders,
    train_one_epoch,
    validate_one_epoch,
)
from filmgrasp.training.losses import TARGET_NAMES
from filmgrasp.training.settings import (
    backbone_init_mode,
    build_run_config,
    dropout_settings,
    loss_settings,
    parse_args,
)

COLUMNS = [
    "epoch",
    "train_loss",
    "train_pos_loss",
    "train_cos_loss",
    "train_sin_loss",
    "train_width_loss",
    "val_loss",
    "val_pos_loss",
    "val_cos_loss",
    "val_sin_loss",
    "val_width_loss",
    "val_grasp_accuracy",
    "val_detection_rate",
    "val_mean_max_iou",
    "learning_rate",
    "epoch_seconds",
]


def build_model(pretrained, device, dropout_enabled=False, dropout_prob=0.0):
    """Backward-compatible model factory exposed from the original module."""
    return _build_model(
        pretrained,
        device,
        dropout_enabled,
        dropout_prob,
        text_encoder_cls=CLIPTextEncoder,
    )


def _logger(path):
    logger = logging.getLogger("filmgrasp.training")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(path, mode="a", encoding="utf-8"),
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_columns(loss_type):
    if loss_type != "masked_geometry":
        return COLUMNS
    diagnostics = [
        "{}_{}_loss".format(split, name)
        for split in ("train", "val")
        for name in A2_DIAGNOSTICS
    ]
    return COLUMNS + diagnostics


def _validate_resume_history(csv_path, start_epoch, columns):
    if not csv_path.is_file():
        raise FileNotFoundError("Missing metrics.csv for resume")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, start_epoch)):
        raise ValueError("CSV epochs do not match checkpoint")
    if rows and set(rows[0]) != set(columns):
        raise ValueError("CSV columns do not match loss configuration")


def _epoch_row(epoch, train_metrics, validation_metrics, learning_rate, seconds):
    row = {
        "epoch": epoch,
        "train_loss": train_metrics["total"],
        "val_loss": validation_metrics["total"],
        "val_grasp_accuracy": validation_metrics["accuracy"],
        "val_detection_rate": validation_metrics["detection_rate"],
        "val_mean_max_iou": validation_metrics["mean_max_iou"],
        "learning_rate": learning_rate,
        "epoch_seconds": seconds,
    }
    for name in TARGET_NAMES:
        row["train_{}_loss".format(name)] = train_metrics[name]
        row["val_{}_loss".format(name)] = validation_metrics[name]
    for name in A2_DIAGNOSTICS:
        if name in train_metrics:
            row["train_{}_loss".format(name)] = train_metrics[name]
            row["val_{}_loss".format(name)] = validation_metrics[name]
    return row


def _write_tensorboard(writer, epoch, row, train_metrics, validation_metrics):
    for name in TARGET_NAMES:
        writer.add_scalar("LossComponents/train_{}".format(name), train_metrics[name], epoch)
        writer.add_scalar(
            "LossComponents/validation_{}".format(name), validation_metrics[name], epoch
        )
    for name in A2_DIAGNOSTICS:
        if name in train_metrics:
            writer.add_scalar("A2/train_{}".format(name), train_metrics[name], epoch)
            writer.add_scalar("A2/validation_{}".format(name), validation_metrics[name], epoch)
    writer.add_scalar("Loss/train", train_metrics["total"], epoch)
    writer.add_scalar("Loss/validation", validation_metrics["total"], epoch)
    writer.add_scalar("Grasp/accuracy", validation_metrics["accuracy"], epoch)
    writer.add_scalar("Grasp/detection_rate", validation_metrics["detection_rate"], epoch)
    writer.add_scalar("Grasp/mean_max_iou", validation_metrics["mean_max_iou"], epoch)
    writer.add_scalar("Optimizer/learning_rate", row["learning_rate"], epoch)


def _is_improved(metrics, best_accuracy, best_loss):
    return metrics["accuracy"] > best_accuracy or (
        metrics["accuracy"] == best_accuracy and metrics["total"] < best_loss
    )


def _save_run_metadata(run_dir, config, data_summary):
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2), encoding="utf-8"
    )


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, validation_loader, data_summary = create_dataloaders(args)

    run_dir = args.output_root / args.run_name
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(
            "Run already exists; choose another run-name or use --resume: {}".format(run_dir)
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir / "train.log")

    config = build_run_config(args, device)
    model = build_model(
        args.pretrained,
        device,
        config["dropout_enabled"],
        config["dropout_prob"],
    )
    optimizer = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.film.parameters()),
        lr=args.lr,
    )

    start_epoch, best_accuracy, best_loss = 1, -1.0, float("inf")
    if args.resume:
        saved = load_checkpoint(args.resume, model, optimizer, config, data_summary)
        start_epoch = saved["epoch"] + 1
        best_accuracy = saved["best_val_accuracy"]
        best_loss = saved["best_val_loss"]
        logger.info("Resuming after epoch %d", saved["epoch"])

    _save_run_metadata(run_dir, config, data_summary)
    csv_path = run_dir / "metrics.csv"
    columns = _metric_columns(args.loss_type)
    if args.resume:
        _validate_resume_history(csv_path, start_epoch, columns)

    tensorboard = SummaryWriter(str(run_dir / "tensorboard"), purge_step=start_epoch)
    loss_kwargs = (
        {"loss_type": args.loss_type, "lambda_bg": args.lambda_bg}
        if args.loss_type == "masked_geometry"
        else {}
    )
    try:
        mode = "a" if args.resume else "w"
        with csv_path.open(mode, newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=columns)
            if not args.resume:
                csv_writer.writeheader()

            for epoch in range(start_epoch, args.epochs + 1):
                # Epoch-specific order also reproduces a resumed run.
                train_loader.sampler.generator.manual_seed(args.seed + epoch)
                began = time.monotonic()
                train_metrics, train_count = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    **loss_kwargs,
                )
                validation_metrics = validate_one_epoch(
                    model,
                    validation_loader,
                    device,
                    **loss_kwargs,
                )
                row = _epoch_row(
                    epoch,
                    train_metrics,
                    validation_metrics,
                    optimizer.param_groups[0]["lr"],
                    time.monotonic() - began,
                )
                _write_tensorboard(tensorboard, epoch, row, train_metrics, validation_metrics)
                csv_writer.writerow(row)
                handle.flush()
                tensorboard.flush()

                improved = _is_improved(validation_metrics, best_accuracy, best_loss)
                if improved:
                    best_accuracy = validation_metrics["accuracy"]
                    best_loss = validation_metrics["total"]
                checkpoint_dir = run_dir / "checkpoints"
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    epoch,
                    best_accuracy,
                    best_loss,
                    config,
                    data_summary,
                )
                if improved:
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        epoch,
                        best_accuracy,
                        best_loss,
                        config,
                        data_summary,
                    )

                logger.info(
                    "epoch=%d train=%d val=%d train_loss=%.6f val_loss=%.6f "
                    "accuracy=%.2f%% detection=%.2f%% iou=%.4f",
                    epoch,
                    train_count,
                    validation_metrics["evaluated_samples"],
                    train_metrics["total"],
                    validation_metrics["total"],
                    validation_metrics["accuracy"],
                    validation_metrics["detection_rate"],
                    validation_metrics["mean_max_iou"],
                )
    finally:
        tensorboard.close()
    return run_dir


if __name__ == "__main__":
    main()
