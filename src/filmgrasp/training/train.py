"""Train and validate FiLM-GraspNet on manifest-based datasets."""

import argparse
import csv
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import evaluate_grasp
from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.film_grasp import FiLMGraspNet
from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.training.losses import TARGET_NAMES, grasp_loss

COLUMNS = ["epoch", "train_loss", "train_pos_loss", "train_cos_loss", "train_sin_loss",
           "train_width_loss", "val_loss", "val_pos_loss", "val_cos_loss", "val_sin_loss",
           "val_width_loss", "val_grasp_accuracy", "val_detection_rate", "val_mean_max_iou",
           "learning_rate", "epoch_seconds"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("../storage/data/train_debug"))
    parser.add_argument("--val-root", type=Path, default=Path("../storage/data/val_debug"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="filmgrasp_debug01")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--pretrained", type=Path, default=Path("checkpoints/grconvnet3_pretrained.pth"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; num-workers must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def _collate(samples):
    batch = {
        "image": torch.stack([s["image"] for s in samples]),
        "prompt": [s["prompt"] for s in samples],
        "targets": {name: torch.stack([s["targets"][name] for s in samples]) for name in TARGET_NAMES},
        "sample_id": [s["sample_id"] for s in samples],
    }
    if "annotations" in samples[0]:
        batch["annotations"] = [s["annotations"] for s in samples]
    return batch


def _summary(dataset, root):
    ids = [s["sample_id"] for s in dataset.samples]
    scenes = sorted({s["scene_id"] for s in dataset.samples})
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return {"root": str(root.resolve()), "samples": len(ids), "scenes": len(scenes),
            "sample_ids": ids, "scene_ids": scenes,
            "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "split": manifest.get("metadata", {}).get("split_file"),
            "provenance": manifest.get("metadata", {})}


def create_dataloaders(args):
    for root in (args.train_root, args.val_root):
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError("Missing dataset manifest: {}".format(root / "manifest.json"))
    train_data = GraspAnythingPPDataset(args.train_root)
    val_data = GraspAnythingPPDataset(args.val_root, include_annotations=True)
    train_ids = {s["scene_id"] for s in train_data.samples}
    val_ids = {s["scene_id"] for s in val_data.samples}
    if train_ids & val_ids:
        raise ValueError("Training and validation scenes overlap")
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=_collate, generator=generator)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, collate_fn=_collate)
    return train, val, {"train": _summary(train_data, args.train_root),
                        "validation": _summary(val_data, args.val_root)}


def build_model(pretrained, device):
    backbone = GRConvNet3(input_channels=3, output_channels=1, channel_size=32, dropout=False)
    if not pretrained.is_file():
        raise FileNotFoundError("Missing pretrained backbone: {}".format(pretrained))
    backbone.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)
    encoder = CLIPTextEncoder(model_name="ViT-B/32", device="cpu")
    model = FiLMGraspNet(backbone, encoder)
    # CLIP stays on CPU to fit the 4 GB GPU.
    model.backbone.to(device)
    model.film.to(device)
    return model


def _prepare(batch, device):
    images = batch["image"].to(device)
    # Dataset already scales RGB to [0,1]; center each image independently.
    images = images - images.mean(dim=(1, 2, 3), keepdim=True)
    targets = {key: value.to(device) for key, value in batch["targets"].items()}
    return images, targets


def _loss_totals():
    return {key: 0.0 for key in ("total",) + TARGET_NAMES}


def _add_loss(totals, loss, components, count):
    totals["total"] += loss.item() * count
    for key in TARGET_NAMES:
        totals[key] += components[key].item() * count


def _averages(totals, count):
    if count == 0:
        raise ValueError("No samples processed")
    return {key: value / count for key, value in totals.items()}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    totals, count = _loss_totals(), 0
    for batch in loader:
        images, targets = _prepare(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, batch["prompt"])
        loss, components = grasp_loss(predictions, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss for {}".format(batch["sample_id"]))
        loss.backward()
        optimizer.step()
        n = images.shape[0]
        _add_loss(totals, loss, components, n)
        count += n
    return _averages(totals, count), count


def aggregate_grasps(results):
    if not results:
        raise ValueError("No validation samples evaluated")
    n = len(results)
    # A missing detection contributes zero IoU and remains in the denominator.
    return {"accuracy": 100.0 * sum(r["success"] for r in results) / n,
            "detection_rate": 100.0 * sum(r["detected"] for r in results) / n,
            "mean_max_iou": sum(r["max_iou"] for r in results) / n,
            "evaluated_samples": n}


def validate_one_epoch(model, loader, device):
    model.eval()
    totals, count, results = _loss_totals(), 0, []
    with torch.no_grad():
        for batch in loader:
            images, targets = _prepare(batch, device)
            predictions = model(images, batch["prompt"])
            loss, components = grasp_loss(predictions, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite validation loss for {}".format(batch["sample_id"]))
            n = images.shape[0]
            _add_loss(totals, loss, components, n)
            for i, annotations in enumerate(batch["annotations"]):
                if annotations.ndim != 2 or annotations.shape[1] != 6 or len(annotations) == 0:
                    raise ValueError("Invalid ground truth for {}".format(batch["sample_id"][i]))
                single = tuple(head[i] for head in predictions)
                try:
                    results.append(evaluate_grasp(single, annotations, threshold=0.25))
                except Exception as exc:
                    raise ValueError("Grasp evaluation failed for {}".format(batch["sample_id"][i])) from exc
            count += n
    metrics = _averages(totals, count)
    metrics.update(aggregate_grasps(results))
    return metrics


def _checkpoint(model, optimizer, epoch, best_accuracy, best_loss, config, data_summary):
    return {"epoch": epoch, "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()
             if not k.startswith("text_encoder.")},
            "optimizer_state_dict": optimizer.state_dict(), "best_val_accuracy": best_accuracy,
            "best_val_loss": best_loss, "config": config, "data_summary": data_summary,
            "rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer, epoch, best_accuracy, best_loss, config, data_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(_checkpoint(model, optimizer, epoch, best_accuracy, best_loss, config, data_summary), temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, config, data_summary):
    if not path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(path))
    
    saved = torch.load(path, map_location="cpu")
    old = saved.get("config", {})
    
    for key in ("pretrained_sha256", "batch_size", "lr", "seed", "optimizer", "clip_model"):
        if old.get(key) != config.get(key):
            raise ValueError("Resume configuration mismatch: {}".format(key))
    
    for split in ("train", "validation"):
        if saved["data_summary"][split]["manifest_sha256"] != data_summary[split]["manifest_sha256"]:
            raise ValueError("Resume dataset mismatch: {}".format(split))
    
    if saved["epoch"] < 1 or saved["epoch"] >= config["epochs"]:
        raise ValueError("Invalid resume epoch {} for {} epochs".format(saved["epoch"], config["epochs"]))
    
    missing, unexpected = model.load_state_dict(saved["model_state_dict"], strict=False)
    
    if unexpected or any(not key.startswith("text_encoder.") for key in missing):
        raise ValueError("Incompatible model checkpoint: missing={} unexpected={}".format(missing, unexpected))
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    torch.set_rng_state(saved["rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    random.setstate(saved["python_rng_state"])
    if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    return saved


def _logger(path):
    logger = logging.getLogger("filmgrasp.training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
           
    device = torch.device(args.device)
    
    train, val, data_summary = create_dataloaders(args)
    
    run_dir = args.output_root / args.run_name
    if run_dir.exists() and args.resume is None:
        raise FileExistsError("Run already exists; choose another run-name or use --resume: {}".format(run_dir))
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    
    logger = _logger(run_dir / "train.log")
    config = {"model": "FiLMGraspNet", "backbone": "GRConvNet3 RGB 32 channels dropout=False",
              "clip_model": "ViT-B/32", "clip_device": "cpu", "input_resolution": [416, 416],
              "optimizer": "Adam", "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
              "num_workers": args.num_workers, "seed": args.seed, "device": str(device),
              "train_root": str(args.train_root), "val_root": str(args.val_root),
              "pretrained": str(args.pretrained),
              "pretrained_sha256": hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "numpy_version": np.__version__}
    model = build_model(args.pretrained, device)
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.film.parameters()), lr=args.lr)
    start, best_accuracy, best_loss = 1, -1.0, float("inf")
    
    
    if args.resume:
        saved = load_checkpoint(args.resume, model, optimizer, config, data_summary)
        start = saved["epoch"] + 1
        best_accuracy, best_loss = saved["best_val_accuracy"], saved["best_val_loss"]
        logger.info("Resuming after epoch %d", saved["epoch"])
        
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
    csv_path = run_dir / "metrics.csv"
    

    if args.resume:
        if not csv_path.is_file():
            raise FileNotFoundError("Missing metrics.csv for resume")
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if [int(row["epoch"]) for row in rows] != list(range(1, start)):
            raise ValueError("CSV epochs do not match checkpoint")
        
    writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=start)
    
    with csv_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if not args.resume:
            csv_writer.writeheader()
            
        for epoch in range(start, args.epochs + 1):
            # Epoch-specific order also reproduces a resumed run
            train.sampler.generator.manual_seed(args.seed + epoch)
            
            began = time.monotonic()
            train_metrics, train_count = train_one_epoch(model, train, optimizer, device)
            val_metrics = validate_one_epoch(model, val, device)
            
            row = {"epoch": epoch, "train_loss": train_metrics["total"],
                   "val_loss": val_metrics["total"], "val_grasp_accuracy": val_metrics["accuracy"],
                   "val_detection_rate": val_metrics["detection_rate"],
                   "val_mean_max_iou": val_metrics["mean_max_iou"],
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "epoch_seconds": time.monotonic() - began}
            
            for name in TARGET_NAMES:
                row["train_{}_loss".format(name)] = train_metrics[name]
                row["val_{}_loss".format(name)] = val_metrics[name]
                writer.add_scalar("LossComponents/train_{}".format(name), train_metrics[name], epoch)
                writer.add_scalar("LossComponents/validation_{}".format(name), val_metrics[name], epoch)
                
            writer.add_scalar("Loss/train", train_metrics["total"], epoch)
            writer.add_scalar("Loss/validation", val_metrics["total"], epoch)
            writer.add_scalar("Grasp/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("Grasp/detection_rate", val_metrics["detection_rate"], epoch)
            writer.add_scalar("Grasp/mean_max_iou", val_metrics["mean_max_iou"], epoch)
            writer.add_scalar("Optimizer/learning_rate", row["learning_rate"], epoch)
            
            csv_writer.writerow(row)
            file.flush()
            writer.flush()
            
            improved = (val_metrics["accuracy"] > best_accuracy or
                        (val_metrics["accuracy"] == best_accuracy and val_metrics["total"] < best_loss))
            if improved:
                best_accuracy, best_loss = val_metrics["accuracy"], val_metrics["total"]
                
            checkpoint_dir = run_dir / "checkpoints"
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch,
                            best_accuracy, best_loss, config, data_summary)
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch,
                                best_accuracy, best_loss, config, data_summary)
                
            logger.info("epoch=%d train=%d val=%d train_loss=%.6f val_loss=%.6f accuracy=%.2f%% detection=%.2f%% iou=%.4f",
                        epoch, train_count, val_metrics["evaluated_samples"], train_metrics["total"],
                        val_metrics["total"], val_metrics["accuracy"], val_metrics["detection_rate"],
                        val_metrics["mean_max_iou"])
    writer.close()
    return run_dir


if __name__ == "__main__":
    main()
