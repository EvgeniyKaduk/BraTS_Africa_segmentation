# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class FastResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1), bias=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(out_ch) if norm else nn.Identity()
        self.bn2 = nn.BatchNorm3d(out_ch) if norm else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch!=out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        out = self.relu(x + identity)
        return out

class FastDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
        )

        self.conv = FastResidualBlock(out_ch + skip_ch, out_ch, norm=False)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FastBottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv3d(ch, ch, 3, padding=4, dilation=4)
        self.fuse  = nn.Conv3d(ch*3, ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        return x + self.fuse(self.relu(torch.cat([c1,c2,c3],1)))

class FastResidualUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_c=32):
        super().__init__()
        self.out_channels = num_classes

        self.enc1 = FastResidualBlock(in_channels, base_c)
        self.enc2 = FastResidualBlock(base_c, base_c*2)
        self.enc3 = FastResidualBlock(base_c*2, base_c*4)
        self.enc4 = FastResidualBlock(base_c*4, base_c*8)

        self.pool = nn.MaxPool3d((1,2,2))

        self.bottleneck = nn.Sequential(
        FastResidualBlock(base_c*8, base_c*12),
        FastResidualBlock(base_c*12, base_c*12),
        FastBottleneck(base_c*12),
        nn.Dropout3d(0.15)
        )


        self.dec4 = FastDecoderBlock(base_c*12, base_c*8, base_c*6)
        self.dec3 = FastDecoderBlock(base_c*6, base_c*4, base_c*4)
        self.dec2 = FastDecoderBlock(base_c*4, base_c*2, base_c*2)
        self.dec1 = FastDecoderBlock(base_c*2, base_c, base_c)

        self.out_conv = nn.Conv3d(base_c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out_conv(d1)

        return out