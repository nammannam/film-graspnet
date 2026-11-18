
"""Smoke test for grasp evaluation metrics."""

import numpy as np
import torch

from filmgrasp.evaluation.grasp_metrics import (
    decode_grasp,
    evaluate_grasp,
)


def main():
    size = 416
    center = 208
    grasp_length = 70.0

    yy, xx = np.mgrid[:size, :size]

    # A smooth quality peak at the expected grasp center.
    q = np.exp(
        -((xx - center)**2 + (yy - center)**2)
        / (2.0 * 8.0**2)
    ).astype(np.float32)

    pos = torch.from_numpy(q)[None, None]

    # Theta = 0: cos(2 theta)=1, sin(2 theta)=0.
    cos = torch.ones_like(pos)
    sin = torch.zeros_like(pos)

    # Ground-truth length is 70 pixels.
    width = torch.full_like(
        pos,
        grasp_length / (size / 2.0),
    )

    predictions = (pos, cos, sin, width)

    # [unused, x, y, length, width, theta_degrees]
    annotations = torch.tensor(
        [[0.0, 208.0, 208.0, 70.0, 35.0, 0.0]],
        dtype=torch.float32,
    )

    grasp = decode_grasp(predictions)

    assert grasp is not None

    print("Predicted center:", grasp.center)
    print("Predicted angle:", grasp.angle)
    print("Predicted length:", grasp.length)

    assert np.allclose(
        grasp.center,
        [208, 208],
        atol=1,
    )

    assert abs(grasp.angle) < 1e-5
    assert abs(grasp.length - grasp_length) < 1.0

    result = evaluate_grasp(
        predictions,
        annotations,
    )

    print("Evaluation:", result)

    assert result["detected"]
    assert result["success"]
    assert result["max_iou"] > 0.9

    # No positive quality peak => no detection.
    empty_predictions = (
        torch.zeros_like(pos),
        cos,
        sin,
        width,
    )

    empty_result = evaluate_grasp(
        empty_predictions,
        annotations,
    )

    assert not empty_result["detected"]
    assert not empty_result["success"]

    print("GRASP METRICS TEST PASSED")


if __name__ == "__main__":
    main()
