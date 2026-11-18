
import torch

from filmgrasp.models.film import FiLMLayer


def main():
    torch.manual_seed(42)

    batch_size = 2
    channels = 128

    film = FiLMLayer(
        text_dim=512,
        feature_channels=channels,
    )

    # Simulated GR-ConvNet visual features.
    features = torch.randn(2, 128, 52, 52)

    # Simulated CLIP text embeddings.
    text_embeddings = torch.randn(2, 512)

    output = film(features, text_embeddings)

    print("Input shape:", features.shape)
    print("Output shape:", output.shape)

    assert output.shape == features.shape
    assert torch.isfinite(output).all()

    # FiLM initially behaves like an identity layer.
    assert torch.allclose(
        output,
        features,
        atol=1e-6,
    )

    print("Identity initialization: OK")

    # Verify that FiLM can receive gradients.
    loss = output.square().mean()
    loss.backward()

    assert film.condition.weight.grad is not None
    assert film.condition.bias.grad is not None

    assert torch.isfinite(
        film.condition.weight.grad
    ).all()

    assert torch.isfinite(
        film.condition.bias.grad
    ).all()

    assert film.condition.weight.grad.abs().sum() > 0

    print("Gradient flow: OK")

    # Confirm that an optimizer update can change FiLM.
    optimizer = torch.optim.Adam(
        film.parameters(),
        lr=1e-3,
    )

    optimizer.step()

    updated_output = film(features, text_embeddings)

    assert not torch.allclose(
        updated_output,
        features,
    )

    print("Optimizer update: OK")
    print("FILM LAYER TEST PASSED")


if __name__ == "__main__":
    main()
