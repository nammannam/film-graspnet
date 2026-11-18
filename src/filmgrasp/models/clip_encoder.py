
"""
Frozen OpenAI CLIP text encoder for FiLM-GraspNet.

Input:
    List[str] of language instructions

Output:
    Float32 tensor [B, 512] for ViT-B/32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cpu",
    ):
        super().__init__()

        self.device_name = torch.device(device)

        self.clip_model, _ = clip.load(
            model_name,
            device=self.device_name,
            jit=False,
        )

        # Freeze the entire pretrained CLIP model.
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        self.embedding_dim = (
            self.clip_model.text_projection.shape[1]
        )

    def train(self, mode=True):
        """
        Keep frozen CLIP in evaluation mode even when
        the parent FiLM-GraspNet calls model.train().
        """
        super().train(False)
        return self

    def forward(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("Expected non-empty prompts")

        tokens = clip.tokenize(
            prompts,
            truncate=True,
        ).to(next(self.clip_model.parameters()).device)

        with torch.no_grad():
            embeddings = self.clip_model.encode_text(tokens)

            # Convert to float32 for FiLM MLP.
            embeddings = embeddings.float()

            # Normalize embeddings to unit length.
            embeddings = F.normalize(
                embeddings,
                dim=-1,
            )

        return embeddings
