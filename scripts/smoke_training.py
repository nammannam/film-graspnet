
"""One-step training test for FiLM-GraspNet."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.film_grasp import FiLMGraspNet


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "storage/data/subset_smoke"
CHECKPOINT = ROOT / "checkpoints/grconvnet3_pretrained.pth"

TARGET_NAMES = ("pos", "cos", "sin", "width")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu"
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device(args.device)

    dataset = GraspAnythingPPDataset(DATA_ROOT)
    sample = dataset[0]

    # Dataset currently returns RGB in [0, 1].
    # Match original Grasp-Anything preprocessing.
    image = sample["image"].unsqueeze(0)
    image = image - image.mean()
    image = image.to(device)

    prompt = [sample["prompt"]]

    targets = [
        sample["targets"][name].unsqueeze(0).to(device)
        for name in TARGET_NAMES
    ]

    backbone = GRConvNet3(
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
    )

    state_dict = torch.load(
        CHECKPOINT, map_location="cpu"
    )
    backbone.load_state_dict(state_dict, strict=True)

    text_encoder = CLIPTextEncoder(
        model_name="ViT-B/32",
        device="cpu",
    )

    model = FiLMGraspNet(backbone, text_encoder)

    # Keep frozen CLIP on CPU to save GPU memory.
    model.backbone.to(device)
    model.film.to(device)

    model.train()

    assert not model.text_encoder.clip_model.training
    assert all(
        not p.requires_grad
        for p in model.text_encoder.parameters()
    )

    optimizer = torch.optim.Adam(
        list(model.backbone.parameters())
        + list(model.film.parameters()),
        lr=1e-4,
    )

    print("Device:", device)
    print("Image:", tuple(image.shape))
    print("Prompt:", prompt[0])

    optimizer.zero_grad(set_to_none=True)

    predictions = model(image, prompt)

    losses = {}

    for name, predicted, target in zip(
        TARGET_NAMES, predictions, targets
    ):
        assert predicted.shape == target.shape, name
        assert torch.isfinite(predicted).all(), name

        losses[name] = F.smooth_l1_loss(
            predicted, target
        )

    total_loss = sum(losses.values())

    assert torch.isfinite(total_loss).item()

    for name, value in losses.items():
        print(f"{name}_loss: {value.item():.6f}")

    print(f"total_loss: {total_loss.item():.6f}")

    before = model.film.condition.weight.detach().clone()

    total_loss.backward()

    grad = model.film.condition.weight.grad

    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0

    print("FiLM gradient: OK")

    optimizer.step()

    after = model.film.condition.weight.detach()

    change = (after - before).abs().max().item()

    assert change > 0
    assert torch.isfinite(after).all()

    print(f"FiLM max weight change: {change:.10f}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory: {peak / 1024**2:.1f} MiB")

    print("FILM-GRASPNET TRAINING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
