
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon


IMAGE_ID = (
    "a0b47d6f55998d10816d86e8b62f51896c4eccfbc7ceab5f2c8b691f6587378f"
)

image_path = Path("data_preview") / f"{IMAGE_ID}.jpg"

# Ground truth extracted from the official annotation
grasp_label = np.array([
    0.097075,
    370.52,
    220.03,
    116.89,
    23.963,
    32.565
], dtype=np.float32)


def grasp_to_corners(label):
    _, x, y, w, h, theta_deg = label

    # Match the LGD angle conversion convention
    theta = -np.deg2rad(theta_deg)

    # Rectangle corners around the origin
    corners = np.array([
        [-w / 2, -h / 2],
        [ w / 2, -h / 2],
        [ w / 2,  h / 2],
        [-w / 2,  h / 2]
    ])

    # Rotation in image coordinates
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    corners = corners @ rotation.T
    corners += np.array([x, y])

    return corners


image = Image.open(image_path).convert("RGB")
corners = grasp_to_corners(grasp_label)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image)

polygon = Polygon(
    corners,
    closed=True,
    fill=False,
    edgecolor="lime",
    linewidth=2.5
)

ax.add_patch(polygon)

ax.scatter(
    grasp_label[1],
    grasp_label[2],
    c="red",
    s=45
)

ax.set_title("Lift apple by its skin.")
ax.set_xlim(0, image.width)
ax.set_ylim(image.height, 0)
ax.axis("off")

output_path = Path("data_preview") / "grasp_visualization.png"
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.show()

print("Saved:", output_path)
print("Corners:\n", corners)
