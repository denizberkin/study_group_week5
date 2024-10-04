import torch
import torch.nn as nn

from .blocks import UpSampleBlock, DownSampleBlock, ConvBlock


class UNet(nn.Module):
    """
    U-Net implementation for the Inzva deep learning study group class.
    paper: https://arxiv.org/pdf/1505.04597

    Args:
        in_channels (int): number of input channels (1 for grayscale, 3 for RGB)
        out_channels (int): number of output channels/classes to predict.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # encoder
        self.maxpool = nn.MaxPool2d(2)
        self.encoder1 = DownSampleBlock(in_channels, 64)
        self.encoder2 = DownSampleBlock(64, 128)
        self.encoder3 = DownSampleBlock(128, 256)
        self.encoder4 = DownSampleBlock(256, 512)

        # bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # decoder
        self.upconv1 = UpSampleBlock(1024, 512)
        self.upconv2 = UpSampleBlock(512, 256)
        self.upconv3 = UpSampleBlock(256, 128)
        self.upconv4 = UpSampleBlock(128, 64)

        # n_channel output
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1, x = self.encoder1(x)
        x2, x = self.encoder2(x)
        x3, x = self.encoder3(x)
        x4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.upconv1(x, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)

        x = self.out(x)
        return x



if __name__ == "__main__":
    device = torch.device("cuda")
    model = UNet(in_channels=1, out_channels=1).to(device)
    input_random_tensor = torch.randn(1, 1, 572, 572).to(device)
    out = model(input_random_tensor)

    print(f"input tensor shape {input_random_tensor.shape}")
    print(f"output tensor shape {out.shape}")
