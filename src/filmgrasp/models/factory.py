"""Construction of the project model with checkpoint-compatible defaults."""

import torch

from filmgrasp.models.clip_encoder import CLIPTextEncoder
from filmgrasp.models.film_grasp import FiLMGraspNet
from filmgrasp.models.grconvnet3 import GRConvNet3
from filmgrasp.training.settings import dropout_settings


def build_model(
    pretrained,
    device,
    dropout_enabled=False,
    dropout_prob=0.0,
    text_encoder_cls=CLIPTextEncoder,
):
    dropout_enabled, dropout_prob = dropout_settings(
        {"dropout_enabled": dropout_enabled, "dropout_prob": dropout_prob}
    )
    backbone = GRConvNet3(
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=dropout_enabled,
        prob=dropout_prob,
    )
    if pretrained is not None:
        if not pretrained.is_file():
            raise FileNotFoundError("Missing pretrained backbone: {}".format(pretrained))
        backbone.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)

    encoder = text_encoder_cls(model_name="ViT-B/32", device="cpu")
    model = FiLMGraspNet(backbone, encoder)
    # CLIP stays on CPU to fit the 4 GB GPU.
    model.backbone.to(device)
    model.film.to(device)
    return model
