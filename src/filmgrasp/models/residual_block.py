import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by GR-ConvNet3."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("ResidualBlock requires equal input and output channels")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        residual = F.relu(self.bn1(self.conv1(x_in)))
        residual = self.bn2(self.conv2(residual))
        return residual + x_in
