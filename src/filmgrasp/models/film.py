
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, text_dim=512, feature_channels=128):
        super().__init__()

        self.feature_channels = feature_channels

        # Predict gamma and beta from CLIP embeddings.
        self.condition = nn.Linear(
            text_dim,
            2 * feature_channels,
        )

        # Initialize FiLM as the identity transformation.
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, features, text_embeddings):
        if features.ndim != 4:
            raise ValueError("Expected features [B,C,H,W]")

        batch_size, channels, _, _ = features.shape

        if channels != self.feature_channels:
            raise ValueError("Unexpected feature channels")

        if text_embeddings.shape != (
            batch_size,
            self.condition.in_features,
        ):
            raise ValueError("Unexpected text embedding shape")

        # [B, 2C]
        params = self.condition(text_embeddings)

        # Each has shape [B, C].
        gamma, beta = params.chunk(2, dim=1)

        # Broadcast over spatial dimensions.
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * features + beta
