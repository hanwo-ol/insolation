# modules_1sobel_DA/unet_ch6_da_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary components from your existing modules
from .double_conv import DoubleConv
from .attention import CBAM                     # <--- Import CBAM
from .attention_da import DualAttentionModule # Import the DA module

class UNet_DA_CBAM_Selective(nn.Module): # Renamed class
    def __init__(self, n_channels=6, n_classes=1, da_reduction=16, use_checkpoint=True):
        super(UNet_DA_CBAM_Selective, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)       # Out: 64
        self.cbam_enc1 = CBAM(64)                   # <--- CBAM after inc
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))    # Out: 128
        # self.cbam_enc2 = CBAM(128)                 # Optional: CBAM here
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))   # Out: 256
        # self.cbam_enc3 = CBAM(256)                 # Optional: CBAM here
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   # Out: 512
        self.cbam_enc4 = CBAM(512)                   # <--- CBAM after down3
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))  # Out: 1024

        # --- Decoder ---
        # DA applied only to deeper levels (x4, x3) where feature maps are smaller

        # Level 1 (Deepest)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Up from bottleneck -> 512
        self.da1 = DualAttentionModule(512, reduction_factor=da_reduction, use_checkpoint=use_checkpoint) # DA for skip connection x4
        self.conv1 = DoubleConv(1024, 512) # Input: 512(skip) + 512(up)
        # self.cbam_dec1 = CBAM(512)                 # Optional: CBAM here

        # Level 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Up from d1 -> 256
        self.da2 = DualAttentionModule(256, reduction_factor=da_reduction, use_checkpoint=use_checkpoint) # DA for skip connection x3
        self.conv2 = DoubleConv(512, 256)  # Input: 256(skip) + 256(up)
        self.cbam_dec2 = CBAM(256)                   # <--- CBAM after conv2

        # Level 3 - NO DA applied to x2
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Up from d2 -> 128
        self.conv3 = DoubleConv(256, 128)  # Input: 128(skip x2) + 128(up)
        # self.cbam_dec3 = CBAM(128)                 # Optional: CBAM here

        # Level 4 - NO DA applied to x1
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Up from d3 -> 64
        self.conv4 = DoubleConv(128, 64)   # Input: 64(skip x1) + 64(up)
        self.cbam_dec4 = CBAM(64)                   # <--- CBAM after conv4

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)            # 64 channels
        x1_att = self.cbam_enc1(x1) # Apply CBAM to x1 features

        x2 = self.down1(x1_att)     # 128 channels (Downsample CBAM'd x1)
        # x2_att = self.cbam_enc2(x2) # Optional

        x3 = self.down2(x2)         # 256 channels
        # x3_att = self.cbam_enc3(x3) # Optional

        x4 = self.down3(x3)         # 512 channels
        x4_att = self.cbam_enc4(x4) # Apply CBAM to x4 features

        x5 = self.down4(x4_att)     # 1024 channels (Downsample CBAM'd x4 - Bottleneck)

        # --- Decoder ---
        # Level 1
        d1 = self.up1(x5)           # Upsample bottleneck: 512 ch
        skip4 = self.da1(x4)        # Apply DA to original x4 skip connection (before encoder CBAM)
        d1 = torch.cat([skip4, d1], dim=1) # Concatenate DA'd skip4 + d1 = 1024 ch
        d1 = self.conv1(d1)         # Process combined -> 512 ch
        # d1_att = self.cbam_dec1(d1) # Optional

        # Level 2
        d2 = self.up2(d1)           # Upsample d1: 256 ch
        skip3 = self.da2(x3)        # Apply DA to original x3 skip connection
        d2 = torch.cat([skip3, d2], dim=1) # Concatenate DA'd skip3 + d2 = 512 ch
        d2 = self.conv2(d2)         # Process combined -> 256 ch
        d2_att = self.cbam_dec2(d2) # Apply CBAM after decoder conv2

        # Level 3 (NO DA)
        d3 = self.up3(d2_att)       # Upsample CBAM'd d2: 128 ch
        # Use original x2 skip connection directly
        d3 = torch.cat([x2, d3], dim=1) # Concatenate original x2 + d3 = 256 ch
        d3 = self.conv3(d3)         # Process combined -> 128 ch
        # d3_att = self.cbam_dec3(d3) # Optional

        # Level 4 (NO DA)
        d4 = self.up4(d3)           # Upsample d3: 64 ch
        # Use original x1 skip connection directly
        d4 = torch.cat([x1, d4], dim=1) # Concatenate original x1 + d4 = 128 ch
        d4 = self.conv4(d4)         # Process combined -> 64 ch
        d4_att = self.cbam_dec4(d4) # Apply CBAM after decoder conv4

        # Final output
        logits = self.outc(d4_att)  # Output from CBAM'd d4
        return logits