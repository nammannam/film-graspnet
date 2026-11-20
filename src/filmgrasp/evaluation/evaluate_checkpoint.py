"""Evaluate a trusted FiLM-GraspNet checkpoint on named datasets."""

import argparse
import csv
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import decode_grasp, evaluate_grasp
from filmgrasp.models.factory import build_model
from filmgrasp.training.checkpoints import validate_evaluation_checkpoint
from filmgrasp.training.engine import (
    collate_samples,
    prepare_batch,
    validate_one_epoch,
)
from filmgrasp.training.settings import backbone_init_mode
from filmgrasp.utils import file_sha256


def parse_dataset(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            "Expected NAME=PATH, e.g. train=../storage/data/train_pilot"
        )
    return name, Path(path).expanduser()


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate FiLM-GraspNet without training.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'last', or a checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Override backbone weights for pretrained runs; omit for random-init runs",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=20)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument(
        "--visualization-selection",
        choices=("random", "failures", "successes"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-samples", nargs=2, metavar=("SAMPLE_A", "SAMPLE_B"))
    parser.add_argument("--paired-output", type=Path)
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or num workers")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    names = [name for name, _ in args.dataset]
    if len(names) != len(set(names)):
        parser.error("Duplicate dataset names")
    if args.visualize and args.max_visualizations < 1:
        parser.error("--max-visualizations must be positive")
    invalid_names = any(Path(name).name != name or name in (".", "..") for name in names)
    if args.visualize and invalid_names:
        parser.error("Visualization dataset names must be single directory names")
    if args.paired_samples and (len(args.dataset) != 1 or args.visualize or args.output):
        parser.error(
            "Paired mode requires one dataset and cannot combine with --visualize or --output"
        )
    if args.paired_output and not args.paired_samples:
        parser.error("--paired-output requires --paired-samples")
    return args


def _checkpoint_path(args):
    if args.checkpoint in ("best", "last"):
        return args.run_dir / "checkpoints" / (args.checkpoint + ".pt")
    return Path(args.checkpoint)


def _resolve_pretrained(args, config):
    init_mode = backbone_init_mode(config)
    if init_mode == "random":
        if args.pretrained is not None:
            raise ValueError("--pretrained cannot be used with a random-initialized run")
        return None
    pretrained = args.pretrained or Path("checkpoints/grconvnet3_pretrained.pth")
    expected_hash = config.get("pretrained_sha256")
    if expected_hash and file_sha256(pretrained) != expected_hash:
        raise ValueError("Pretrained backbone SHA-256 mismatch")
    return pretrained


def _restore_model(checkpoint, pretrained, device, dropout):
    model = build_model(pretrained, device, *dropout)
    compatibility = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing = list(compatibility.missing_keys)
    unexpected = list(compatibility.unexpected_keys)
    invalid_missing = [
        name for name in missing if not name.startswith("text_encoder.clip_model.")
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "Model state mismatch: missing={} unexpected={}".format(
                invalid_missing, unexpected
            )
        )
    model.eval()
    return model, missing


def install_numpy_compat():
    """Support trusted checkpoints serialized with NumPy 2.x."""
    if np.__version__.split(".")[0] != "1":
        return

    import numpy.core.multiarray
    import numpy.core.numeric
    import numpy.core._multiarray_umath

    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core._multiarray_umath"] = (
        np.core._multiarray_umath
    )


def load_checkpoint(path):
    """Only use with checkpoints created by this trusted project."""
    install_numpy_compat()

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    return torch.load(path, **kwargs)


def check_dataset_provenance(checkpoint, name, root):
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    summaries = checkpoint.get("data_summary", {})
    expected = summaries.get(name, {}).get("manifest_sha256")

    if expected is None:
        print("  Provenance: no matching checkpoint record")
        return

    actual = file_sha256(manifest)
    if actual != expected:
        raise ValueError(
            "Dataset manifest hash mismatch for {}:\n"
            "  expected: {}\n"
            "  actual:   {}".format(name, expected, actual)
        )

    print("  Provenance: manifest SHA-256 MATCH")


class VisualizationCollector:
    def __init__(self, selection, limit, seed):
        self.selection = selection
        self.limit = limit
        self.rng = random.Random(seed)
        self.candidate_count = 0
        self.entries = []

    def __call__(self, batch, index, heads, result):
        if self.selection == "failures" and result["success"]:
            return
        if self.selection == "successes" and not result["success"]:
            return
        self.candidate_count += 1
        priority = self.rng.random()
        if len(self.entries) == self.limit and priority >= self.entries[-1][0]:
            return
        prediction = decode_grasp(heads)
        if (prediction is not None) != result["detected"]:
            raise AssertionError("Prediction and evaluator detection disagree")
        image = batch["image"][index].permute(1, 2, 0).numpy().copy()
        quality = heads[0].detach().float().cpu().squeeze().numpy().copy()
        sample = {
            "sample_id": batch["sample_id"][index],
            "prompt": batch["prompt"][index],
            "scene_id": batch["scene_id"][index],
            "annotations": batch["annotations"][index],
            "image": image,
            "quality": quality,
            "prediction": prediction,
            "result": result.copy(),
        }
        self.entries.append((priority, sample))
        self.entries.sort(key=lambda item: item[0])
        if len(self.entries) > self.limit:
            self.entries.pop()


def save_visualizations(collector, directory, name, root, metrics, seed):
    from filmgrasp.evaluation.visualize_checkpoint import (
        COLUMNS, make_contact_sheet, render_figure,
    )

    figures = directory / "predictions"
    figures.mkdir(parents=True, exist_ok=False)
    rows, paths = [], []
    for _, sample in collector.entries:
        prediction = sample["prediction"]
        result = sample["result"]
        relative = Path("predictions") / (sample["sample_id"] + ".png")
        path = directory / relative
        render_figure(path, sample["image"], sample["prompt"], sample["sample_id"],
                      sample["annotations"], prediction, result, sample["quality"], 8)
        row = {"sample_id": sample["sample_id"], "scene_id": sample["scene_id"],
               "prompt": sample["prompt"], "detected": result["detected"],
               "success": result["success"], "max_iou": result["max_iou"],
               "pred_center_row": None, "pred_center_col": None, "pred_angle_deg": None,
               "pred_length_px": None, "pred_width_px": None, "figure_path": str(relative)}
        if prediction is not None:
            row.update(pred_center_row=float(prediction.center[0]),
                       pred_center_col=float(prediction.center[1]),
                       pred_angle_deg=float(np.rad2deg(prediction.angle)),
                       pred_length_px=float(prediction.length),
                       pred_width_px=float(prediction.width))
        rows.append(row)
        paths.append(path)
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(paths, directory / "contact_sheet.png")
    metadata = {"dataset": name, "root": str(root.resolve()),
                "selection": collector.selection, "seed": seed,
                "candidate_count": collector.candidate_count,
                "selected_sample_ids": [row["sample_id"] for row in rows],
                "rendered_count": len(rows), "evaluated_samples": metrics["evaluated_samples"]}
    (directory / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("Visualization candidates:", collector.candidate_count)
    print("Selected sample IDs:", metadata["selected_sample_ids"])
    print("Saved visualizations:", directory)
    return metadata


def visualize_instruction_pair(model, root, sample_ids, device, output):
    """Evaluate and render two prompts for the same scene and RGB tensor."""
    from filmgrasp.evaluation.visualize_checkpoint import render_instruction_pair

    dataset = GraspAnythingPPDataset(root, include_annotations=True)
    indices = {record["sample_id"]: index for index, record in enumerate(dataset.samples)}
    if len(set(sample_ids)) != 2 or any(sample_id not in indices for sample_id in sample_ids):
        raise ValueError("Expected two distinct sample IDs present in the dataset")
    samples = [dataset[indices[sample_id]] for sample_id in sample_ids]
    if samples[0]["scene_id"] != samples[1]["scene_id"]:
        raise ValueError("Paired samples belong to different scenes")
    if not torch.equal(samples[0]["image"], samples[1]["image"]):
        raise ValueError("Paired samples have different RGB tensors")

    rendered = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            images, _ = prepare_batch(collate_samples([sample]), device)
            heads = model(images, [sample["prompt"]])
            single = tuple(head[0] for head in heads)
            result = evaluate_grasp(single, sample["annotations"])
            prediction = decode_grasp(single)
            if (prediction is not None) != result["detected"]:
                raise AssertionError("Prediction and evaluator detection disagree")
            rendered.append({"image": sample["image"].permute(1, 2, 0).numpy(),
                             "prompt": sample["prompt"], "annotations": sample["annotations"],
                             "prediction": prediction, "result": result})
            center = None if prediction is None else [int(v) for v in prediction.center]
            print("{} | {} | center(row,col)={} | max_iou={:.6f} | {}".format(
                sample["sample_id"], sample["prompt"], center, result["max_iou"],
                "SUCCESS" if result["success"] else "FAILURE"))
    output.parent.mkdir(parents=True, exist_ok=True)
    render_instruction_pair(output, rendered)
    print("Shared scene:", samples[0]["scene_id"])
    print("Identical RGB tensors: True")
    print("Saved:", output)


def main(argv=None):
    args = parse_args(argv)
    names = [name for name, _ in args.dataset]
    checkpoint_path = _checkpoint_path(args)
    visualization_base = (
        args.visualization_dir
        or args.run_dir / "visualizations" / checkpoint_path.stem
    )
    if args.visualize:
        for name in names:
            if (visualization_base / name).exists():
                raise FileExistsError("Refusing to overwrite: {}".format(visualization_base / name))

    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pretrained = _resolve_pretrained(args, config)
    checkpoint = load_checkpoint(checkpoint_path)
    saved_dropout, saved_loss = validate_evaluation_checkpoint(checkpoint, config)

    print("Checkpoint:", checkpoint_path)
    print("Epoch:", checkpoint["epoch"])
    print("Device:", args.device)
    print("Building model...")

    model, missing = _restore_model(checkpoint, pretrained, args.device, saved_dropout)
    print("Backbone/FiLM state: OK")
    print("Frozen CLIP tensors omitted:", len(missing))

    if args.paired_samples:
        name, root = args.dataset[0]
        check_dataset_provenance(checkpoint, name, root)
        output = args.paired_output or (
            args.run_dir / "visualizations" / "instruction_sensitivity" / "pair.png"
        )
        visualize_instruction_pair(model, root, args.paired_samples, args.device, output)
        return

    results = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": args.device,
        "training_loss_type": saved_loss[0],
        "training_lambda_bg": saved_loss[1],
        "evaluation_loss_type": "b0",
        "datasets": {},
    }

    for name, root in args.dataset:
        print("\n=== {} ===".format(name.upper()))
        check_dataset_provenance(checkpoint, name, root)

        dataset = GraspAnythingPPDataset(
            root, include_annotations=True
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_samples,
        )

        collector = (VisualizationCollector(args.visualization_selection,
                                            args.max_visualizations, args.seed)
                     if args.visualize else None)
        if collector is None:
            metrics = validate_one_epoch(model, loader, args.device)
        else:
            metrics = validate_one_epoch(model, loader, args.device,
                                         sample_callback=collector)

        results["datasets"][name] = {
            "root": str(root.resolve()),
            "samples": len(dataset),
            "manifest_sha256": file_sha256(
                root / "manifest.json"
            ),
            "metrics": metrics,
        }
        if collector is not None:
            results["datasets"][name]["visualization"] = save_visualizations(
                collector, visualization_base / name, name, root, metrics, args.seed)

        print("Samples:", metrics["evaluated_samples"])
        print("Loss:", round(metrics["total"], 6))
        print("Accuracy: {:.2f}%".format(metrics["accuracy"]))
        print("Detection: {:.2f}%".format(
            metrics["detection_rate"]
        ))
        print("Mean max IoU: {:.6f}".format(
            metrics["mean_max_iou"]
        ))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(
                "Refusing to overwrite: {}".format(args.output)
            )
        args.output.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
