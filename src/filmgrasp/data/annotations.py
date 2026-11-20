
"""Parsing and validation helpers for Grasp-Anything++ annotations."""

import inspect
import io

import torch


def parse_sample_id(filename):
    """Extract scene, object and part identifiers.

    Expected filename form: ``<scene_hash>_<object>_<part>.pkl``.
    """
    name = filename.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]

    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None

    scene_id, object_id, part_id = parts

    if not object_id.isdigit() or not part_id.isdigit():
        return None

    return {
        "scene_id": scene_id,
        "sample_id": stem,
        "object_id": object_id,
        "part_id": part_id,
    }


def _validate_grasp_tensor(grasps):
    if not isinstance(grasps, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    if grasps.ndim != 2 or grasps.shape[1] != 6:
        raise ValueError("Invalid grasp shape: {}".format(tuple(grasps.shape)))
    if len(grasps) == 0:
        raise ValueError("Empty grasp annotation")
    if not torch.isfinite(grasps).all().item():
        raise ValueError("Non-finite grasp values")
    if not (grasps[:, 3:5] > 0).all().item():
        raise ValueError("Invalid width or height")
    return grasps


def load_grasp_tensor(source):
    """Load and validate a tensor from a trusted path or binary stream."""
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return _validate_grasp_tensor(torch.load(source, **kwargs))


def validate_positive_grasps(raw_bytes):
    """Validate serialized Grasp-Anything++ labels and return their row count.

    The input is trusted project/dataset content. It must deserialize to a
    finite tensor shaped ``[N, 6]`` with positive width and height.
    """
    return len(load_grasp_tensor(io.BytesIO(raw_bytes)))
