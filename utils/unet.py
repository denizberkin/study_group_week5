import torch
import torch.nn as nn

from blocks import UpSampleBlock, DownSampleBlock, ConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.ds1 = DownSampleBlock(in_channels, 64)
        self.ds2 = DownSampleBlock(64, 128)
        self.ds3 = DownSampleBlock(128, 256)
        self.ds4 = DownSampleBlock(256, 512)

        self.conv = ConvBlock(512, 1024, padding=1)

        self.us1 = UpSampleBlock(1024, 512)
        self.us2 = UpSampleBlock(512, 256)
        self.us3 = UpSampleBlock(256, 128)
        self.us4 = UpSampleBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual1, x = self.ds1(x)
        residual2, x = self.ds2(x)
        residual3, x = self.ds3(x)
        residual4, x = self.ds4(x)

        conv_out = self.conv(x)

        print(conv_out.shape, residual4.shape)
        x = self.us1(conv_out, residual4)
        print(x.shape, residual3.shape)
        x = self.us2(x, residual3)
        x = self.us3(x, residual2)
        x = self.us4(x, residual1)

        return self.out(x)


if __name__ == "__main__":

    model = UNet(in_channels=1, out_channels=1)
    input_random_tensor = torch.randn(1, 1, 256, 256)
    print(input_random_tensor.shape)
    out = model(input_random_tensor)

    print(f"input tensor shape {input_random_tensor.shape}")
    print(f"output tensor shape {out.shape}")
