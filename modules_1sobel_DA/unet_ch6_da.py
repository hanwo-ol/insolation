# modules_1sobel_DA/unet_ch6_da.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary components from your existing modules
# Make sure the relative path is correct for your project structure
from .double_conv import DoubleConv
from .attention_da import DualAttentionModule

class UNet_DA_Selective(nn.Module): # Renamed class to reflect selective DA
    def __init__(self, n_channels=6, n_classes=1, da_reduction=16, use_checkpoint=True):
        super(UNet_DA_Selective, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)       # Out: 64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))    # Out: 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))   # Out: 256
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   # Out: 512
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))  # Out: 1024

        # --- Decoder ---
        # DA applied only to deeper levels (x4, x3) where feature maps are smaller

        # Level 1 (Deepest)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Up from bottleneck -> 512 channels
        self.da1 = DualAttentionModule(512, reduction_factor=da_reduction, use_checkpoint=use_checkpoint) # DA for skip connection x4 (512 channels)
        # Input to conv1: 512 (from skip4=da1(x4)) + 512 (from up1) = 1024
        self.conv1 = DoubleConv(1024, 512) # Output: 512

        # Level 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Up from d1 -> 256 channels
        self.da2 = DualAttentionModule(256, reduction_factor=da_reduction, use_checkpoint=use_checkpoint) # DA for skip connection x3 (256 channels)
        # Input to conv2: 256 (from skip3=da2(x3)) + 256 (from up2) = 512
        self.conv2 = DoubleConv(512, 256)  # Output: 256

        # Level 3 - NO DA applied to x2 (128 channels)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Up from d2 -> 128 channels
        # Input to conv3: 128 (from original skip x2) + 128 (from up3) = 256
        self.conv3 = DoubleConv(256, 128)  # Output: 128

        # Level 4 - NO DA applied to x1 (64 channels)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Up from d3 -> 64 channels
        # Input to conv4: 64 (from original skip x1) + 64 (from up4) = 128
        self.conv4 = DoubleConv(128, 64)   # Output: 64

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)        # Output: 64
        x2 = self.down1(x1)     # Output: 128
        x3 = self.down2(x2)     # Output: 256
        x4 = self.down3(x3)     # Output: 512
        x5 = self.down4(x4)     # Output: 1024 (Bottleneck)

        # --- Decoder ---
        # Level 1
        d1 = self.up1(x5)          # Upsample bottleneck: 512 ch
        skip4 = self.da1(x4)       # Apply DA to x4 (512 ch skip)
        d1 = torch.cat([skip4, d1], dim=1) # Concatenate skip4 + d1 = 1024 ch
        d1 = self.conv1(d1)        # Process combined -> 512 ch

        # Level 2
        d2 = self.up2(d1)          # Upsample d1: 256 ch
        skip3 = self.da2(x3)       # Apply DA to x3 (256 ch skip)
        d2 = torch.cat([skip3, d2], dim=1) # Concatenate skip3 + d2 = 512 ch
        d2 = self.conv2(d2)        # Process combined -> 256 ch

        # Level 3 (NO DA)
        d3 = self.up3(d2)          # Upsample d2: 128 ch
        # Use original x2 skip connection directly
        d3 = torch.cat([x2, d3], dim=1) # Concatenate x2 + d3 = 256 ch
        d3 = self.conv3(d3)        # Process combined -> 128 ch

        # Level 4 (NO DA)
        d4 = self.up4(d3)          # Upsample d3: 64 ch
        # Use original x1 skip connection directly
        d4 = torch.cat([x1, d4], dim=1) # Concatenate x1 + d4 = 128 ch
        d4 = self.conv4(d4)        # Process combined -> 64 ch

        # Final output
        logits = self.outc(d4)
        return logits