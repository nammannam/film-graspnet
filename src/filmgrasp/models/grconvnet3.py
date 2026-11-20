"""RGB GR-ConvNet3 backbone with separable encoder and decoder stages."""

import torch.nn as nn
import torch.nn.functional as F

from filmgrasp.models.residual_block import ResidualBlock


class GRConvNet3(nn.Module):
    """Checkpoint-compatible GR-ConvNet3 used by FiLM-GraspNet."""

    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
        prob=0.0,
    ):
        super().__init__()
        if not isinstance(dropout, bool):
            raise TypeError("dropout must be a boolean")
        if not 0 <= prob < 1:
            raise ValueError("dropout probability must satisfy 0 <= p < 1")

        encoded_channels = channel_size * 4
        self.conv1 = nn.Conv2d(
            input_channels, channel_size, kernel_size=9, stride=1, padding=4
        )
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(
            channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.conv3 = nn.Conv2d(
            channel_size * 2, encoded_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(encoded_channels)

        self.res1 = ResidualBlock(encoded_channels, encoded_channels)
        self.res2 = ResidualBlock(encoded_channels, encoded_channels)
        self.res3 = ResidualBlock(encoded_channels, encoded_channels)
        self.res4 = ResidualBlock(encoded_channels, encoded_channels)
        self.res5 = ResidualBlock(encoded_channels, encoded_channels)

        self.conv4 = nn.ConvTranspose2d(
            encoded_channels,
            channel_size * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn4 = nn.BatchNorm2d(channel_size * 2)
        self.conv5 = nn.ConvTranspose2d(
            channel_size * 2,
            channel_size,
            kernel_size=4,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.ConvTranspose2d(
            channel_size, channel_size, kernel_size=9, stride=1, padding=4
        )

        self.pos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight, gain=1)

    def encode(self, images):
        features = F.relu(self.bn1(self.conv1(images)))
        features = F.relu(self.bn2(self.conv2(features)))
        return F.relu(self.bn3(self.conv3(features)))

    def decode(self, features):
        features = self.res1(features)
        features = self.res2(features)
        features = self.res3(features)
        features = self.res4(features)
        features = self.res5(features)
        features = F.relu(self.bn4(self.conv4(features)))
        features = F.relu(self.bn5(self.conv5(features)))
        features = self.conv6(features)

        if self.dropout:
            pos = self.pos_output(self.dropout_pos(features))
            cos = self.cos_output(self.dropout_cos(features))
            sin = self.sin_output(self.dropout_sin(features))
            width = self.width_output(self.dropout_wid(features))
        else:
            pos = self.pos_output(features)
            cos = self.cos_output(features)
            sin = self.sin_output(features)
            width = self.width_output(features)
        return pos, cos, sin, width

    def forward(self, images):
        return self.decode(self.encode(images))
