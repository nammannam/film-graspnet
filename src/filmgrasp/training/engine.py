"""Data loading and single-epoch training/validation routines."""

import hashlib
import json

import torch
from torch.utils.data import DataLoader

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import evaluate_grasp
from filmgrasp.training.losses import TARGET_NAMES, grasp_loss

A2_DIAGNOSTICS = tuple(
    "{}_{}".format(name, region)
    for name in TARGET_NAMES[1:]
    for region in ("positive", "background_weighted")
)


def collate_samples(samples):
    """Collate manifest samples while keeping prompts and annotations as lists."""
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "prompt": [sample["prompt"] for sample in samples],
        "targets": {
            name: torch.stack([sample["targets"][name] for sample in samples])
            for name in TARGET_NAMES
        },
        "sample_id": [sample["sample_id"] for sample in samples],
        "scene_id": [sample["scene_id"] for sample in samples],
    }
    if "annotations" in samples[0]:
        batch["annotations"] = [sample["annotations"] for sample in samples]
    return batch


def prepare_batch(batch, device):
    """Move tensors to the compute device and apply image centering."""
    images = batch["image"].to(device)
    images = images - images.mean(dim=(1, 2, 3), keepdim=True)
    targets = {name: value.to(device) for name, value in batch["targets"].items()}
    return images, targets


# Backward-compatible private names used by existing notebooks/tests.
_collate = collate_samples
_prepare = prepare_batch


def _dataset_summary(dataset, root):
    sample_ids = [sample["sample_id"] for sample in dataset.samples]
    scene_ids = sorted({sample["scene_id"] for sample in dataset.samples})
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "root": str(root.resolve()),
        "samples": len(sample_ids),
        "scenes": len(scene_ids),
        "sample_ids": sample_ids,
        "scene_ids": scene_ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": manifest.get("metadata", {}).get("split_file"),
        "provenance": manifest.get("metadata", {}),
    }


def create_dataloaders(args):
    for root in (args.train_root, args.val_root):
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError("Missing dataset manifest: {}".format(manifest))

    train_data = GraspAnythingPPDataset(args.train_root)
    validation_data = GraspAnythingPPDataset(args.val_root, include_annotations=True)
    train_scenes = {sample["scene_id"] for sample in train_data.samples}
    validation_scenes = {sample["scene_id"] for sample in validation_data.samples}
    if train_scenes & validation_scenes:
        raise ValueError("Training and validation scenes overlap")

    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_samples,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_data, shuffle=False, **common)
    summaries = {
        "train": _dataset_summary(train_data, args.train_root),
        "validation": _dataset_summary(validation_data, args.val_root),
    }
    return train_loader, validation_loader, summaries


def _loss_totals(loss_type="b0"):
    names = ("total",) + TARGET_NAMES
    if loss_type == "masked_geometry":
        names += A2_DIAGNOSTICS
    return {name: 0.0 for name in names}


def _add_loss(totals, loss, components, count, diagnostics=None):
    totals["total"] += loss.item() * count
    for name in TARGET_NAMES:
        totals[name] += components[name].item() * count
    if diagnostics is not None:
        for name in A2_DIAGNOSTICS:
            totals[name] += diagnostics[name].item() * count


def _averages(totals, count):
    if count == 0:
        raise ValueError("No samples processed")
    return {name: value / count for name, value in totals.items()}


def _checked_loss(predictions, targets, loss_type, lambda_bg, diagnostics, batch):
    loss, components = grasp_loss(
        predictions,
        targets,
        loss_type,
        lambda_bg,
        diagnostics,
    )
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nonfinite loss for {}".format(batch["sample_id"]))
    return loss, components


def train_one_epoch(model, loader, optimizer, device, loss_type="b0", lambda_bg=0.05):
    model.train()
    totals = _loss_totals(loss_type)
    diagnostics = {} if loss_type == "masked_geometry" else None
    count = 0
    for batch in loader:
        images, targets = prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, batch["prompt"])
        loss, components = _checked_loss(
            predictions, targets, loss_type, lambda_bg, diagnostics, batch
        )
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        _add_loss(totals, loss, components, batch_size, diagnostics)
        count += batch_size
    return _averages(totals, count), count


def aggregate_grasps(results):
    if not results:
        raise ValueError("No validation samples evaluated")
    count = len(results)
    # A missing detection contributes zero IoU and remains in the denominator.
    return {
        "accuracy": 100.0 * sum(result["success"] for result in results) / count,
        "detection_rate": 100.0 * sum(result["detected"] for result in results) / count,
        "mean_max_iou": sum(result["max_iou"] for result in results) / count,
        "evaluated_samples": count,
    }


def _validate_annotations(annotations, sample_id):
    if annotations.ndim != 2 or annotations.shape[1] != 6 or len(annotations) == 0:
        raise ValueError("Invalid ground truth for {}".format(sample_id))


def validate_one_epoch(
    model,
    loader,
    device,
    sample_callback=None,
    loss_type="b0",
    lambda_bg=0.05,
):
    model.eval()
    totals = _loss_totals(loss_type)
    diagnostics = {} if loss_type == "masked_geometry" else None
    count, results = 0, []
    with torch.no_grad():
        for batch in loader:
            images, targets = prepare_batch(batch, device)
            predictions = model(images, batch["prompt"])
            loss, components = _checked_loss(
                predictions, targets, loss_type, lambda_bg, diagnostics, batch
            )
            batch_size = images.shape[0]
            _add_loss(totals, loss, components, batch_size, diagnostics)
            for index, annotations in enumerate(batch["annotations"]):
                sample_id = batch["sample_id"][index]
                _validate_annotations(annotations, sample_id)
                heads = tuple(head[index] for head in predictions)
                try:
                    result = evaluate_grasp(heads, annotations, threshold=0.25)
                except Exception as exc:
                    raise ValueError("Grasp evaluation failed for {}".format(sample_id)) from exc
                results.append(result)
                if sample_callback is not None:
                    sample_callback(batch, index, heads, result)
            count += batch_size

    metrics = _averages(totals, count)
    metrics.update(aggregate_grasps(results))
    return metrics
