
"""
Visualize ground-truth grasp rectangles and positive target
regions over RGB images.

Geometry follows the LGD Grasp representation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.geometry.grasp import Grasp


DATA_ROOT = Path("../storage/data/subset_smoke")
OUTPUT_DIR = Path("data_preview/targets")


def plot_sample(dataset, index):
    sample = dataset[index]
    record = dataset.samples[index]

    image = sample["image"].permute(1, 2, 0).numpy()
    pos = sample["targets"]["pos"][0].numpy()

    # Read original grasp labels.
    grasps = torch.load(
        DATA_ROOT / record["grasp_path"],
        map_location="cpu",
    ).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Original RGB image.
    ax.imshow(image)

    # 2. Positive grasp regions.
    # Mask the background to keep the image visible.
    mask = np.ma.masked_where(pos <= 0, pos)

    ax.imshow(
        mask,
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.38,
        interpolation="nearest",
    )

    # 3. Full grasp rectangles and centers.
    for row in grasps:
        _, x, y, length, width, theta_deg = row

        angle = -np.deg2rad(theta_deg)

        grasp = Grasp(
            center=np.array([y, x]),
            angle=angle,
            length=length,
            width=width,
        )

        # LGD stores polygon points in [row, col].
        corners = grasp.as_gr.points[:, [1, 0]]

        polygon = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor="#12A89D",
            linewidth=1.4,
            alpha=0.9,
        )

        ax.add_patch(polygon)

        ax.plot(
            x, y,
            marker="o",
            color="#D94848",
            markersize=3,
        )

    # 4. Formatting.
    ax.set_title(
        sample["prompt"],
        fontsize=11,
        pad=10,
    )

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_items = [
        Line2D(
            [0], [0],
            color="#12A89D",
            linewidth=2,
            label="GT grasp rectangle",
        ),
        mpatches.Patch(
            facecolor="#E5A32A",
            alpha=0.5,
            label="Positive target region",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="#D94848",
            markersize=5,
            label="Grasp center",
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.text(
        0.5, 0.085,
        f"Ground truth | {len(grasps)} grasps",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.91,
        bottom=0.17,
    )

    output_path = (
        OUTPUT_DIR / f"gt_overlay_{index:02d}.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        facecolor="white",
    )

    plt.close(fig)
    print("Saved:", output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = GraspAnythingPPDataset(DATA_ROOT)

    for index in range(len(dataset)):
        plot_sample(dataset, index)

    print(f"Generated {len(dataset)} GT overlays.")


if __name__ == "__main__":
    main()
