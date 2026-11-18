
"""
Convert Grasp-Anything++ positive grasp annotations into
GR-ConvNet target maps using LGD grasp geometry.
"""

import numpy as np
import torch

from filmgrasp.geometry.grasp import (
    Grasp,
    GraspRectangles,
)


def build_grasp_targets(annotations, image_size=(416, 416)):
    """
    Args:
        annotations: Tensor or ndarray of shape [N, 6].
        image_size: (height, width), default (416, 416).

    Returns:
        Dictionary of four float32 tensors:
        pos, cos, sin, width, each [1, H, W].
    """
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.detach().cpu().numpy()

    annotations = np.asarray(annotations, dtype=np.float32)

    if annotations.ndim != 2 or annotations.shape[1] != 6:
        raise ValueError(
            f"Expected annotations [N, 6], got {annotations.shape}"
        )

    if len(annotations) == 0:
        raise ValueError("Grasp annotations are empty")

    if not np.isfinite(annotations).all():
        raise ValueError("Annotations contain NaN or Inf")

    if np.any(annotations[:, 3:5] <= 0):
        raise ValueError("Grasp width/height must be positive")

    height, width = image_size

    if height != width:
        raise ValueError(
            "Current implementation expects a square image"
        )

    rectangles = []

    for row in annotations:
        _, x, y, grasp_length, grasp_height, theta_deg = row

        # LGD convention: center is [row, column] = [y, x].
        center = np.array([y, x], dtype=np.float32)

        # Grasp-Anything degrees -> LGD radians.
        angle = -float(theta_deg) * np.pi / 180.0

        grasp = Grasp(
            center=center,
            angle=angle,
            length=float(grasp_length),
            width=float(grasp_height),
        )

        rectangles.append(grasp.as_gr)

    grasp_rectangles = GraspRectangles(rectangles)

    # Draw the compact central grasp regions using LGD.
    pos_map, angle_map, width_map = grasp_rectangles.draw(
        shape=(height, width)
    )

    # Follow LGD's width normalization.
    width_map = np.clip(
        width_map, 0.0, width / 2.0
    ) / (width / 2.0)

    cos_map = np.cos(2.0 * angle_map)
    sin_map = np.sin(2.0 * angle_map)

    def as_tensor(array):
        array = np.asarray(array, dtype=np.float32)
        return torch.from_numpy(array[None, ...].copy())

    return {
        "pos": as_tensor(pos_map),
        "cos": as_tensor(cos_map),
        "sin": as_tensor(sin_map),
        "width": as_tensor(width_map),
    }
