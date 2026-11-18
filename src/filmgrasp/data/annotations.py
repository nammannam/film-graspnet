
import io
import torch


def parse_sample_id(filename):
    """
    Extract scene, object and part identifiers.

    Example:
    <scene_hash>_0_0.pkl
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


def validate_positive_grasps(raw_bytes):
    """
    Validate Grasp-Anything++ labels [N, 6].

    Returns number of valid grasp rows.
    """
    grasps = torch.load(
        io.BytesIO(raw_bytes),
        map_location="cpu"
    )

    if not isinstance(grasps, torch.Tensor):
        raise TypeError("Expected torch.Tensor")

    if grasps.ndim != 2 or grasps.shape[1] != 6:
        raise ValueError(
            f"Invalid grasp shape: {tuple(grasps.shape)}"
        )

    if len(grasps) == 0:
        raise ValueError("Empty grasp annotation")

    if not torch.isfinite(grasps).all().item():
        raise ValueError("Non-finite grasp values")

    if not (grasps[:, 3:5] > 0).all().item():
        raise ValueError("Invalid width or height")

    return len(grasps)
