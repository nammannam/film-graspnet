"""Check a local GR-ConvNet3 checkpoint and its four output maps.

Only load checkpoints from sources you trust: these upstream files contain
pickled PyTorch model objects, which torch.load deserializes as Python objects.
"""

import argparse
import sys
from pathlib import Path

import torch


ASSIGNMENT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "grasp-anything": (
        "Grasp-Anything",
        "weights/model_grasp_anything",
    ),
    "lgd-rgbd": (
        "LGD",
        "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_42_iou_0.93",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", 
                        choices=CHECKPOINTS, 
                        default="grasp-anything")
    parser.add_argument("--device", 
                        choices=("cpu", "cuda"), 
                        default="cpu")
    args = parser.parse_args()

    repo_name, relative_checkpoint = CHECKPOINTS[args.model]
    repo = ASSIGNMENT_ROOT / "refer_gitrepo" / repo_name
    checkpoint = repo / relative_checkpoint
    if not checkpoint.is_file():
        parser.error("Checkpoint not found: {}".format(checkpoint))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this process")

    # The upstream checkpoint stores a complete model object whose class lives
    # under inference.models.grconvnet3 in its own repository.
    sys.path.insert(0, str(repo))
    model = torch.load(str(checkpoint), map_location=args.device)
    if type(model).__module__ != "inference.models.grconvnet3":
        raise TypeError("Unexpected checkpoint model: {}".format(type(model)))
    model.eval()

    channels = model.conv1.in_channels
    sample = torch.randn(1, channels, 224, 224, device=args.device)
    with torch.no_grad():
        outputs = model(sample)

    names = ("pos", "cos", "sin", "width")
    expected_shape = (1, 1, 224, 224)
    if len(outputs) != len(names):
        raise AssertionError("Expected four output maps, got {}".format(len(outputs)))
    for name, output in zip(names, outputs):
        if tuple(output.shape) != expected_shape:
            raise AssertionError("{} shape: {}".format(name, tuple(output.shape)))

    print("Checkpoint:", checkpoint)
    print("Model:", type(model).__module__ + "." + type(model).__name__)
    print("Device:", args.device)
    print("Input:", tuple(sample.shape))
    for name, output in zip(names, outputs):
        print("{}: {}, dtype: {}, finite: {}".format(name, tuple(output.shape), output.dtype, torch.isfinite(output).all().item()))


if __name__ == "__main__":
    main()
