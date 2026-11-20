
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

from filmgrasp.data.annotations import load_grasp_tensor
from filmgrasp.data.targets import build_grasp_targets

IMAGE_SIZE = (416, 416)


class GraspAnythingPPDataset(Dataset):
    def __init__(self, root, include_annotations=False):
        self.root = Path(root)
        self.include_annotations = include_annotations
        manifest_path = self.root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples = data["samples"]
        if not self.samples:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]

        image_path = self.root / record["image_path"]
        grasp_path = self.root / record["grasp_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != IMAGE_SIZE:
                raise ValueError("Expected 416x416, got {}".format(image.size))
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        # copy() makes the transposed array contiguous and writable for PyTorch.
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        grasps = load_grasp_tensor(grasp_path)
        targets = build_grasp_targets(grasps, image_size=IMAGE_SIZE)

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
