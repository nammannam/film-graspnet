"""Export the trusted Grasp-Anything GR-ConvNet3 as a plain state dict."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "refer_gitrepo" / "Grasp-Anything"
SOURCE = REFERENCE / "weights" / "model_grasp_anything"
DESTINATION = ROOT / "checkpoints" / "grconvnet3_pretrained.pth"


def main():
    # The trusted upstream checkpoint stores a pickled model object whose class
    # must be importable while converting it to a safer tensor-only artifact.
    sys.path.insert(0, str(REFERENCE))
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    original = torch.load(SOURCE, map_location="cpu")
    if original.conv1.in_channels != 3:
        raise ValueError("Expected an RGB GR-ConvNet3 checkpoint")

    print("Original class:", type(original))
    print("Input channels:", original.conv1.in_channels)
    print("Channel size:", original.conv1.out_channels)
    print("FiLM channels:", original.conv3.out_channels)

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original.state_dict(), DESTINATION)
    print("Exported:", DESTINATION)
    print("PRETRAINED EXPORT PASSED")


if __name__ == "__main__":
    main()
