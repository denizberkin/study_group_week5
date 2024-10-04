import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        res_conv = self.conv(x)
        res_pool = self.pool(res_conv)

        return res_conv, res_pool  # return residual of conv operation as well 


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # residual shape issues with concat
        dH = skip.size()[2] - x.size()[2]
        dW = skip.size()[3] - x.size()[3]

        x = nn.functional.pad(x, [dW // 2, dW - dW // 2,
                                  dH // 2, dH - dH // 2])

        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
