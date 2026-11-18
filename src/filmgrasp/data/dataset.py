
"""
Manifest-based Grasp-Anything++ dataset for FiLM GraspNet.

Each sample provides:
    - RGB image tensor
    - Language instruction
    - GR-ConvNet grasp target maps
    - Scene/sample identifiers
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from filmgrasp.data.targets import build_grasp_targets


class GraspAnythingPPDataset(Dataset):
    def __init__(self, root, include_annotations=False):
        self.root = Path(root)
        self.include_annotations = include_annotations

        manifest_path = self.root / "manifest.json"

        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.samples = data["samples"]

        if not self.samples:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]

        image_path = self.root / record["image_path"]
        grasp_path = self.root / record["grasp_path"]

        # Read RGB image.
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if image.size != (416, 416):
                raise ValueError(
                    f"Expected 416x416, got {image.size}"
                )

            image_array = np.asarray(
                image, dtype=np.float32
            ) / 255.0

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # Read grasp annotations.
        grasps = torch.load(
            grasp_path,
            map_location="cpu"
        )

        # Convert annotations to dense target maps.
        targets = build_grasp_targets(
            grasps,
            image_size=(416, 416)
        )

        sample = {
            "image": image_tensor,
            "prompt": record["prompt"],
            "targets": targets,
            "scene_id": record["scene_id"],
            "sample_id": record["sample_id"],
        }
        if self.include_annotations:
            sample["annotations"] = grasps
        return sample
