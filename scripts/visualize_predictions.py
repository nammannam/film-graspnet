"""Render real FiLM-GraspNet predictions and original grasp annotations."""

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import torch
from PIL import Image, ImageOps

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import annotations_to_rectangles, decode_grasp, evaluate_grasp
from filmgrasp.training.train import build_model

COLUMNS = ("sample_id", "scene_id", "prompt", "detected", "success", "max_iou",
           "pred_center_row", "pred_center_col", "pred_angle_deg", "pred_length_px",
           "pred_width_px", "figure_path")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("../storage/data/val_debug"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/filmgrasp_cuda_debug01/figures"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--max-gt-to-draw", type=int, default=8)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.max_gt_to_draw < 1:
        parser.error("--max-gt-to-draw must be positive")
    return args


def load_trained_model(path, device):
    """Reconstruct CLIP and restore every saved backbone/FiLM tensor."""
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    if config.get("model") != "FiLMGraspNet" or config.get("clip_model") != "ViT-B/32":
        raise ValueError("Incompatible checkpoint model or CLIP identifier")
    pretrained = Path(config["pretrained"])
    if not pretrained.is_file():
        raise FileNotFoundError("Pretrained backbone required to reconstruct model: {}".format(pretrained))
    model = build_model(pretrained, device)
    saved = checkpoint["model_state_dict"]
    expected = {name for name in model.state_dict() if not name.startswith("text_encoder.")}
    if set(saved) != expected:
        raise ValueError("Checkpoint trainable weights differ: missing={}, unexpected={}".format(
            sorted(expected - set(saved)), sorted(set(saved) - expected)))
    missing, unexpected = model.load_state_dict(saved, strict=False)
    if unexpected or any(not name.startswith("text_encoder.") for name in missing):
        raise ValueError("Checkpoint state is incompatible with FiLM-GraspNet")
    model.eval()
    if any(parameter.requires_grad for parameter in model.text_encoder.parameters()):
        raise AssertionError("CLIP was not frozen")
    if next(model.text_encoder.parameters()).device.type != "cpu":
        raise AssertionError("CLIP must remain on CPU")
    return model, checkpoint


def rectangle_xy(rectangle):
    """Convert existing (row, column) corners to plotting (x, y) corners."""
    points = np.asarray(rectangle.points)
    if points.shape != (4, 2):
        raise ValueError("Expected four grasp rectangle corners")
    return points[:, [1, 0]]


def draw_rectangle(ax, rectangle, color, linewidth=2):
    ax.add_patch(Polygon(rectangle_xy(rectangle), closed=True, fill=False,
                         edgecolor=color, linewidth=linewidth))


def select_gt_for_display(rectangles, prediction, limit):
    """Show the first K GTs, replacing the last with the closest match if needed."""
    best = None
    if prediction is not None and rectangles:
        best = max(range(len(rectangles)), key=lambda i: prediction.as_gr.iou(rectangles[i]))
    indices = list(range(min(limit, len(rectangles))))
    if best is not None and best not in indices:
        indices[-1] = best
    return [rectangles[i] for i in indices], (rectangles[best] if best is not None else None)


def render_figure(path, rgb, prompt, sample_id, annotations, prediction, result,
                  quality_map, max_gt_to_draw):
    """Render a bounded GT subset while evaluation uses every original annotation."""
    rectangles = annotations_to_rectangles(annotations).grs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    display_gt, closest_gt = select_gt_for_display(rectangles, prediction, max_gt_to_draw)
    for ax in axes[:2]:
        ax.imshow(rgb, origin="upper")
        ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Column (px)")
        ax.set_ylabel("Row (px)")
    for rectangle in display_gt:
        draw_rectangle(axes[0], rectangle, "#19a65a", linewidth=1.6)
    axes[0].set_title("A  Original RGB + ground truth\n{} of {} rectangles shown".format(
        len(display_gt), len(rectangles)))
    if closest_gt is not None and result["max_iou"] > 0:
        draw_rectangle(axes[1], closest_gt, "#19a65a", linewidth=1.6)
    if prediction is not None:
        draw_rectangle(axes[1], prediction.as_gr, "#e53935", linewidth=2.4)
    axes[1].set_title("B  Top-1 prediction + closest ground truth")
    heatmap = axes[2].imshow(quality_map, cmap="magma", origin="upper", vmin=0, vmax=1)
    axes[2].set_xlim(-0.5, quality_map.shape[1] - 0.5)
    axes[2].set_ylim(quality_map.shape[0] - 0.5, -0.5)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("Column (px)")
    axes[2].set_ylabel("Row (px)")
    axes[2].set_title("C  Predicted grasp quality (0–1 display scale)")
    if prediction is not None:
        axes[2].plot(prediction.center[1], prediction.center[0], marker="+", markersize=12,
                     markeredgewidth=2, color="cyan")
    fig.colorbar(heatmap, ax=axes[2], shrink=0.77, label="Raw position head (display clipped to 0–1)")
    label = "SUCCESS" if result["success"] else "FAILURE"
    title = "{}  |  IoU {:.4f}  |  {}".format(sample_id, result["max_iou"], label)
    fig.suptitle(title + "\n" + textwrap.fill(prompt, width=100), fontsize=12, fontweight="semibold")
    fig.legend(handles=[Line2D([0], [0], color="#19a65a", lw=2, label="Ground truth"),
                        Line2D([0], [0], color="#e53935", lw=2, label="Top-1 prediction"),
                        Line2D([0], [0], marker="+", color="cyan", linestyle="None",
                               label="Selected center")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def make_contact_sheet(paths, output_path):
    """Create a compact review sheet from the generated figures."""
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            thumbs.append(ImageOps.contain(image, (1200, 500)))
    if not thumbs:
        return
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 1200, rows * 500), "white")
    for i, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((i % columns) * 1200, (i // columns) * 500))
    canvas.save(output_path)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = GraspAnythingPPDataset(args.data_root, include_annotations=True)
    manifest_ids = [record["sample_id"] for record in dataset.samples]
    if args.sample_ids:
        requested = list(dict.fromkeys(args.sample_ids))
        unknown = set(requested) - set(manifest_ids)
        if unknown:
            raise ValueError("Sample IDs absent from manifest: {}".format(sorted(unknown)))
        indices = [manifest_ids.index(sample_id) for sample_id in requested]
    else:
        indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    model, checkpoint = load_trained_model(args.checkpoint, device)
    figures_dir = args.output_dir / "predictions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows, figure_paths = [], []
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            rgb_tensor = sample["image"]
            rgb = rgb_tensor.permute(1, 2, 0).numpy()
            image = rgb_tensor.unsqueeze(0).to(device)
            image = image - image.mean(dim=(1, 2, 3), keepdim=True)
            outputs = model(image, [sample["prompt"]])
            heads = tuple(output[0] for output in outputs)
            prediction = decode_grasp(heads)
            result = evaluate_grasp(heads, sample["annotations"], threshold=0.25)
            if bool(prediction is not None) != result["detected"]:
                raise AssertionError("Prediction and evaluator detection disagree")
            sample_id = sample["sample_id"]
            relative_path = Path("predictions") / (sample_id + ".png")
            figure_path = args.output_dir / relative_path
            render_figure(figure_path, rgb, sample["prompt"], sample_id,
                          sample["annotations"], prediction, result,
                          heads[0].detach().float().cpu().squeeze().numpy(), args.max_gt_to_draw)
            row = {"sample_id": sample_id, "scene_id": sample["scene_id"],
                   "prompt": sample["prompt"], "detected": result["detected"],
                   "success": result["success"], "max_iou": result["max_iou"],
                   "pred_center_row": None, "pred_center_col": None, "pred_angle_deg": None,
                   "pred_length_px": None, "pred_width_px": None,
                   "figure_path": str(relative_path)}
            if prediction is not None:
                row.update(pred_center_row=float(prediction.center[0]),
                           pred_center_col=float(prediction.center[1]),
                           pred_angle_deg=float(np.rad2deg(prediction.angle)),
                           pred_length_px=float(prediction.length),
                           pred_width_px=float(prediction.width))
            rows.append(row)
            figure_paths.append(figure_path)
            print("{} detected={} success={} IoU={:.4f}".format(
                sample_id, result["detected"], result["success"], result["max_iou"]))
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    count = len(rows)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
               "dataset": str(args.data_root.resolve()), "samples_processed": count,
               "successful_grasps": sum(row["success"] for row in rows),
               "detected_grasps": sum(row["detected"] for row in rows),
               "grasp_accuracy_percent": 100 * sum(row["success"] for row in rows) / count,
               "detection_rate_percent": 100 * sum(row["detected"] for row in rows) / count,
               "mean_max_iou": sum(row["max_iou"] for row in rows) / count,
               "evaluation": {"top_k": 1, "iou_threshold": 0.25, "strict_threshold": True,
                              "angular_tolerance_degrees": 30,
                              "no_detection_iou": 0.0,
                              "gt_evaluation": "all original annotations",
                              "gt_display": "first {} annotations in stored order; replace last with closest GT if omitted".format(args.max_gt_to_draw)},
               "selection": "explicit sample IDs" if args.sample_ids else "first samples in manifest order",
               "device": str(device), "seed": args.seed}
    (args.output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(figure_paths, args.output_dir / "contact_sheet.png")
    print(json.dumps({key: summary[key] for key in ("samples_processed", "grasp_accuracy_percent",
                                                     "detection_rate_percent", "mean_max_iou")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
