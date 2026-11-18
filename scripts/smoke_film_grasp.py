
"""FiLM-GraspNet integration smoke test."""

from pathlib import Path

import torch

from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.film_grasp import FiLMGraspNet


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/grconvnet3_pretrained.pth"


def main():
    torch.manual_seed(42)

    # 1. Build standalone pretrained backbone.
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

    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    # 2. Build frozen CLIP encoder.
    text_encoder = CLIPTextEncoder(
        model_name="ViT-B/32",
        device="cpu",
    )

    # 3. Use the same image for both models.
    image = torch.randn(1, 3, 224, 224)
    prompt = ["Grip key on its ring."]

    with torch.no_grad():
        original_outputs = backbone(image)

    # 4. Assemble the proposed model.
    model = FiLMGraspNet(backbone, text_encoder)
    model.eval()

    with torch.no_grad():
        outputs = model(image, prompt)

    assert len(outputs) == 4

    # 5. Compare each grasp prediction head.
    for name, original, predicted in zip(
        ("pos", "cos", "sin", "width"),
        original_outputs,
        outputs,
    ):
        assert predicted.shape == (1, 1, 224, 224)
        assert torch.isfinite(predicted).all()

        max_error = (
            original - predicted
        ).abs().max().item()

        print(f"{name}: max_error={max_error:.10f}")

        assert torch.allclose(
            original,
            predicted,
            atol=1e-5,
            rtol=1e-5,
        ), f"{name} identity mismatch"

    # 6. Ensure CLIP remains frozen.
    assert all(
        not p.requires_grad
        for p in model.text_encoder.parameters()
    )

    # 7. Ensure FiLM is trainable.
    assert all(
        p.requires_grad
        for p in model.film.parameters()
    )

    print("FiLM identity initialization: PASSED")
    print("CLIP frozen: PASSED")
    print("FiLM trainable: PASSED")
    print("FILM-GRASPNET INTEGRATION PASSED")


if __name__ == "__main__":
    main()
