
"""Grasp decoding and evaluation metrics."""

import numpy as np
from scipy.ndimage import gaussian_filter

from filmgrasp.geometry.grasp import (
    Grasp,
    GraspRectangles,
    detect_grasps,
)


def _to_map(tensor):
    """Convert one prediction head into a 2D NumPy map."""
    array = tensor.detach().float().cpu().numpy()
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a single 2D map, got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError("Prediction contains NaN or Inf")

    return array


def decode_grasp(predictions):
    """
    Decode four output maps into one grasp.

    predictions: (pos, cos, sin, width)
    Returns: Grasp or None.
    """
    if len(predictions) != 4:
        raise ValueError("Expected four prediction maps")

    pos, cos, sin, width = map(_to_map, predictions)

    if not (pos.shape == cos.shape == sin.shape == width.shape):
        raise ValueError("Prediction map shapes do not match")

    height, image_width = pos.shape

    if height != image_width:
        raise ValueError(
            "Current width normalization assumes square images"
        )

    # Post-processing.
    q_img = gaussian_filter(pos, sigma=2.0)

    angle_img = 0.5 * np.arctan2(sin, cos)
    angle_img = gaussian_filter(angle_img, sigma=2.0)

    # Reverse the target width normalization.
    length_img = np.clip(width, 0.0, 1.0) * (height / 2.0)
    length_img = gaussian_filter(length_img, sigma=1.0)

    # Original Grasp-Anything peak detector.
    grasps = detect_grasps(
        q_img,
        angle_img,
        width_img=length_img,
        no_grasps=1,
    )

    if not grasps:
        return None

    return grasps[0]


def annotations_to_rectangles(annotations):
    """
    Convert Grasp-Anything++ annotation [N,6] to GT.

    Format:
        [unused, x, y, length, width, theta_degrees]
    """
    if hasattr(annotations, "detach"):
        annotations = annotations.detach().cpu().numpy()

    annotations = np.asarray(annotations)

    if annotations.ndim != 2 or annotations.shape[1] != 6:
        raise ValueError(
            f"Expected annotations [N,6], got {annotations.shape}"
        )

    rectangles = []

    for row in annotations:
        if not np.isfinite(row).all():
            raise ValueError("Invalid GT annotation")

        _, x, y, length, width, theta_deg = row

        grasp = Grasp(
            center=np.array([y, x], dtype=np.float32),
            angle=-np.deg2rad(theta_deg),
            length=float(length),
            width=float(width),
        )

        rectangles.append(grasp.as_gr)

    return GraspRectangles(rectangles)


def evaluate_grasp(predictions, annotations, threshold=0.25):
    """
    Evaluate top-1 predicted grasp against GT rectangles.

    Returns:
        success: bool
        detected: bool
        max_iou: float
    """
    predicted_grasp = decode_grasp(predictions)

    if predicted_grasp is None:
        return {
            "success": False,
            "detected": False,
            "max_iou": 0.0,
        }

    gt_rectangles = annotations_to_rectangles(annotations)

    # Check for empty GT
    if len(gt_rectangles.grs) == 0:
        raise ValueError("No ground-truth grasps")

    max_iou = float(predicted_grasp.max_iou(gt_rectangles))

    return {
        "success": bool(max_iou > threshold),
        "detected": True,
        "max_iou": max_iou,
    }
