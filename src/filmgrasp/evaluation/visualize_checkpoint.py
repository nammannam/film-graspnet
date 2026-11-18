"""Reusable plotting utilities for checkpoint evaluation."""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageOps

from filmgrasp.evaluation.grasp_metrics import annotations_to_rectangles

COLUMNS = ("sample_id", "scene_id", "prompt", "detected", "success", "max_iou",
           "pred_center_row", "pred_center_col", "pred_angle_deg", "pred_length_px",
           "pred_width_px", "figure_path")


def rectangle_xy(rectangle):
    """Convert existing (row, column) corners to plotting (x, y) corners."""
    points = np.asarray(rectangle.points)
    if points.shape != (4, 2):
        raise ValueError("Expected four grasp rectangle corners")
    return points[:, [1, 0]]


def draw_rectangle(ax, rectangle, color, linewidth=2):
    ax.add_patch(Polygon(rectangle_xy(rectangle), closed=True, fill=False,
                         edgecolor=color, linewidth=linewidth))


def render_instruction_pair(path, samples):
    """Draw two instruction-conditioned grasps on an identical RGB image."""
    if len(samples) != 2 or not np.array_equal(samples[0]["image"], samples[1]["image"]):
        raise ValueError("Paired samples must have identical RGB images")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.05))
    for ax, sample in zip(axes, samples):
        rgb = sample["image"]
        ax.imshow(rgb, origin="upper")
        ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        for rectangle in annotations_to_rectangles(sample["annotations"]).grs:
            draw_rectangle(ax, rectangle, "#19a65a", linewidth=1.15)
        prediction = sample["prediction"]
        if prediction is not None:
            draw_rectangle(ax, prediction.as_gr, "#e53935", linewidth=2.0)
        result = sample["result"]
        status = "SUCCESS" if result["success"] else "FAILURE"
        detection = "" if prediction is not None else "  |  No detection"
        ax.set_title(sample["prompt"], fontsize=10, pad=7)
        ax.text(0.5, -0.035, "Max IoU {:.4f}  |  {}{}".format(
            result["max_iou"], status, detection), transform=ax.transAxes,
            ha="center", va="top", fontsize=9)
    fig.legend(handles=[Line2D([0], [0], color="#19a65a", lw=2, label="Ground truth"),
                        Line2D([0], [0], color="#e53935", lw=2, label="Predicted grasp")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.16, wspace=0.04)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


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
    title = "{}  |  detected={}  |  IoU {:.4f}  |  {}".format(
        sample_id, result.get("detected", prediction is not None), result["max_iou"], label)
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
        Image.new("RGB", (1200, 500), "white").save(output_path)
        return
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 1200, rows * 500), "white")
    for i, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((i % columns) * 1200, (i // columns) * 500))
    canvas.save(output_path)
