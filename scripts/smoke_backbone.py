
"""Standalone GR-ConvNet3 smoke test."""

from pathlib import Path

import torch

from filmgrasp.models.grconvnet3 import GRConvNet3


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/grconvnet3_pretrained.pth"


def main():
    torch.manual_seed(42)

    backbone = GRConvNet3(
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
    )

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    backbone.load_state_dict(
        state_dict,
        strict=True,
    )

    backbone.eval()

    images = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = backbone.encode(images)
        outputs = backbone(images)

    print("Encoder:", tuple(features.shape))

    assert features.shape == (1, 128, 56, 56)

    names = ("pos", "cos", "sin", "width")

    assert len(outputs) == 4

    for name, output in zip(names, outputs):
        print(name, tuple(output.shape))

        assert output.shape == (1, 1, 224, 224)
        assert torch.isfinite(output).all()

    print("STRICT WEIGHT LOADING PASSED")
    print("STANDALONE BACKBONE TEST PASSED")


if __name__ == "__main__":
    main()
