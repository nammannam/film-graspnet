
"""Grasp prediction losses."""

import math

import torch.nn.functional as F


TARGET_NAMES = ("pos", "cos", "sin", "width")


def validate_loss_settings(loss_type, lambda_bg):
    if loss_type not in ("b0", "masked_geometry"):
        raise ValueError("Invalid loss type")
    if not isinstance(lambda_bg, (int, float)) or not math.isfinite(lambda_bg) or lambda_bg < 0:
        raise ValueError("lambda_bg must be finite and nonnegative")
    return loss_type, float(lambda_bg)


def grasp_loss(predictions, targets, loss_type="b0", lambda_bg=0.05, diagnostics=None):
    loss_type, lambda_bg = validate_loss_settings(loss_type, lambda_bg)
    if len(predictions) != len(TARGET_NAMES):
        raise ValueError("Expected four grasp predictions")

    components = {}
    if diagnostics is not None:
        diagnostics.clear()
    # A2: use ground-truth position support; retain the B0 pixel denominator.
    mask = (targets["pos"] > 0) if loss_type == "masked_geometry" else None

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

        if loss_type == "b0" or name == "pos":
            components[name] = F.smooth_l1_loss(prediction, target)
        else:
            if mask.shape != target.shape:
                raise ValueError("A2 mask shape mismatch")
            elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
            positive = (elementwise * mask).mean()
            background = (elementwise * (~mask)).mean() * lambda_bg
            components[name] = positive + background
            if diagnostics is not None:
                diagnostics[name + "_positive"] = positive
                diagnostics[name + "_background_weighted"] = background

    total_loss = sum(components.values())

    return total_loss, components
