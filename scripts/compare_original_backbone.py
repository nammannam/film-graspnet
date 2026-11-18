
"""Compare original Grasp-Anything and standalone GR-ConvNet3."""

import sys
from pathlib import Path

import torch

from filmgrasp.models.grconvnet3 import GRConvNet3


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "refer_gitrepo" / "Grasp-Anything"
ORIGINAL_CHECKPOINT = REFERENCE / "weights" / "model_grasp_anything"
LOCAL_CHECKPOINT = ROOT / "checkpoints" / "grconvnet3_pretrained.pth"


def main():
    torch.manual_seed(42)

    # Only needed for the original trusted checkpoint.
    sys.path.insert(0, str(REFERENCE))

    original = torch.load(
        ORIGINAL_CHECKPOINT,
        map_location="cpu",
    )

    local = GRConvNet3(
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
    )

    state_dict = torch.load(
        LOCAL_CHECKPOINT,
        map_location="cpu",
    )
    local.load_state_dict(state_dict, strict=True)

    original.eval()
    local.eval()

    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        original_outputs = original(image)
        local_outputs = local(image)

    for name, expected, actual in zip(
        ("pos", "cos", "sin", "width"),
        original_outputs,
        local_outputs,
    ):
        max_error = (expected - actual).abs().max().item()
        print(f"{name}: max_error={max_error:.10f}")

        assert torch.allclose(
            expected,
            actual,
            atol=1e-5,
            rtol=1e-5,
        ), f"{name} mismatch"

    print("BACKBONE IDENTITY TEST PASSED")


if __name__ == "__main__":
    main()
