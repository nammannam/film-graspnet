
"""Language-Conditioned FiLM-GraspNet."""

import torch
import torch.nn as nn

from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.film import FiLMLayer


class FiLMGraspNet(nn.Module):
    def __init__(self, backbone, text_encoder):
        super().__init__()

        if not isinstance(backbone, GRConvNet3):
            raise TypeError("Expected a GRConvNet3 backbone")

        self.backbone = backbone
        self.text_encoder = text_encoder

        # The CLIP encoder must remain frozen.
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Infer dimensions from the actual modules.
        feature_channels = self.backbone.conv3.out_channels
        text_dim = self.text_encoder.embedding_dim

        self.film = FiLMLayer(
            text_dim=text_dim,
            feature_channels=feature_channels,
        )

    def forward(self, images, prompts):
        if images.ndim != 4:
            raise ValueError("Images must have shape [B,C,H,W]")

        if images.shape[1] != self.backbone.conv1.in_channels:
            raise ValueError("Unexpected number of image channels")

        if isinstance(prompts, str):
            prompts = [prompts]

        if len(prompts) != images.shape[0]:
            raise ValueError(
                "Prompt count must match image batch size"
            )

        # 1. Extract visual features.
        visual_features = self.backbone.encode(images)

        # 2. Extract frozen language features.
        text_features = self.text_encoder(prompts)

        # 3. Transfer embeddings to the visual feature device.
        text_features = text_features.to(
            device=visual_features.device,
            dtype=visual_features.dtype,
        )

        # 4. Condition visual features on language.
        conditioned_features = self.film(
            visual_features,
            text_features,
        )

        # 5. Predict four dense grasp maps.
        return self.backbone.decode(conditioned_features)
