
"""Grasp prediction losses."""

import torch.nn.functional as F


TARGET_NAMES = ("pos", "cos", "sin", "width")


def grasp_loss(predictions, targets):
    if len(predictions) != len(TARGET_NAMES):
        raise ValueError("Expected four grasp predictions")

    components = {}

    for name, prediction in zip(
        TARGET_NAMES,
        predictions,
    ):
        target = targets[name]

        if prediction.shape != target.shape:
            raise ValueError(
                f"{name}: prediction {prediction.shape} "
                f"!= target {target.shape}"
            )

        components[name] = F.smooth_l1_loss(
            prediction,
            target,
        )

    total_loss = sum(components.values())

    return total_loss, components
