# modules/unet.py
import torch
import torch.nn as nn
from modules_1pw.double_conv import DoubleConv  # Import DoubleConv
from modules_1pw.attention import CBAM  # Import CBAM


class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.cbam64 = CBAM(64)
        self.cbam128 = CBAM(128)
        self.cbam512 = CBAM(512)
        self.cbam1024 = CBAM(1024)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down22 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 2048))
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.up1_conv = DoubleConv(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2_conv = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3_conv = DoubleConv(512, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4_conv = DoubleConv(256, 128)
        self.up44 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up44_conv = DoubleConv(128, 64)
        self.up5 = CBAM(64)  # You can remove this if you're not using it
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.cbam64(x1)
        x3 = self.down2(x2)
        x33 = self.down22(x3)
        x4 = self.down3(x33)
        x44 = self.cbam512(x4)
        x5 = self.down4(x44)
        x6 = self.down5(x5)
        x = self.up1(x6)
        x = self.up1_conv(torch.cat([x5, x], dim=1))
        x = self.up2(x)
        x = self.up2_conv(torch.cat([x44, x], dim=1))
        x = self.cbam512(x)
        x = self.up3(x)
        x = self.up3_conv(torch.cat([x33, x], dim=1))
        x = self.up4(x)
        x = self.up4_conv(torch.cat([x3, x], dim=1))
        x = self.up44(x)
        x = self.up44_conv(torch.cat([x2, x], dim=1))
        x = self.cbam64(x)
        return self.outc(x)