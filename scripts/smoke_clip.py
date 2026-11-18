
"""
Smoke test for frozen CLIP text encoder.
"""

import torch

from filmgrasp.models.clip_encoder import CLIPTextEncoder


def main():
    encoder = CLIPTextEncoder(
        model_name="ViT-B/32",
        device="cpu",
    )

    prompts = [
        "Grip key on its ring.",
        "Grip notebook on its binding.",
    ]

    embeddings = encoder(prompts)

    print("Prompts:", len(prompts))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()

    assert all(
        not parameter.requires_grad
        for parameter in encoder.parameters()
    )

    assert not embeddings.requires_grad

    # Check deterministic output in eval mode.
    embeddings_2 = encoder(prompts)

    assert torch.allclose(
        embeddings,
        embeddings_2,
        atol=1e-5,
    )

    # Different instructions should generate
    # different text embeddings.
    assert not torch.allclose(
        embeddings[0],
        embeddings[1],
    )

    # Parent model may call train().
    # Frozen CLIP must stay in eval mode.
    encoder.train()

    assert not encoder.clip_model.training

    print("CLIP frozen:", True)
    print("CLIP TEXT ENCODER TEST PASSED")


if __name__ == "__main__":
    main()
