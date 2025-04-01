# modules_1sobel_DA/unet_ch6_parametric.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary components from your existing modules
from .double_conv import DoubleConv
from .attention import CBAM                     # Import CBAM
from .attention_da import DualAttentionModule # Import the DA module

class UNet_Single_DA_CBAM(nn.Module): # Renamed class
    """
    U-Net with CBAM modules and selectively applied Dual Attention (DA)
    on a single specified skip connection level.
    """
    def __init__(self, n_channels=6, n_classes=1, da_reduction=16, use_checkpoint=True, da_level=None):
        """
        Initializes the UNet.

        Args:
            n_channels (int): Number of input image channels.
            n_classes (int): Number of output classes.
            da_reduction (int): Channel reduction factor for DA modules.
            use_checkpoint (bool): Whether to use gradient checkpointing for DA modules during training.
            da_level (int, optional): The level (1-4) at which to apply DA.
                                       1 applies to deepest skip (512 ch),
                                       2 applies to next skip (256 ch),
                                       3 applies to next skip (128 ch),
                                       4 applies to shallowest skip (64 ch).
                                       None or 0 applies no DA. Defaults to None.
        """
        super(UNet_Single_DA_CBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.da_level = da_level if da_level in [1, 2, 3, 4] else None # Ensure da_level is valid or None

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)       # Out: 64
        self.cbam_enc1 = CBAM(64)                   # CBAM after inc
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))    # Out: 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))   # Out: 256
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   # Out: 512
        self.cbam_enc4 = CBAM(512)                   # CBAM after down3
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))  # Out: 1024

        # --- DA Modules (Instantiate all, apply selectively) ---
        if self.da_level is not None:
             self.da1 = DualAttentionModule(512, reduction_factor=da_reduction, use_checkpoint=use_checkpoint)
             self.da2 = DualAttentionModule(256, reduction_factor=da_reduction, use_checkpoint=use_checkpoint)
             self.da3 = DualAttentionModule(128, reduction_factor=da_reduction, use_checkpoint=use_checkpoint)
             self.da4 = DualAttentionModule(64, reduction_factor=da_reduction, use_checkpoint=use_checkpoint)

        # --- Decoder ---
        # Level 1 (Deepest)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Up from bottleneck -> 512
        self.conv1 = DoubleConv(1024, 512) # Input: 512(skip) + 512(up)

        # Level 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Up from d1 -> 256
        self.conv2 = DoubleConv(512, 256)  # Input: 256(skip) + 256(up)
        self.cbam_dec2 = CBAM(256)                   # CBAM after conv2

        # Level 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Up from d2 -> 128
        self.conv3 = DoubleConv(256, 128)  # Input: 128(skip x2) + 128(up)

        # Level 4
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Up from d3 -> 64
        self.conv4 = DoubleConv(128, 64)   # Input: 64(skip x1) + 64(up)
        self.cbam_dec4 = CBAM(64)                   # CBAM after conv4

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)            # 64 channels
        x1_att = self.cbam_enc1(x1) # Apply CBAM to x1 features

        x2 = self.down1(x1_att)     # 128 channels (Downsample CBAM'd x1)
        x3 = self.down2(x2)         # 256 channels
        x4 = self.down3(x3)         # 512 channels
        x4_att = self.cbam_enc4(x4) # Apply CBAM to x4 features

        x5 = self.down4(x4_att)     # 1024 channels (Downsample CBAM'd x4 - Bottleneck)

        # --- Decoder ---
        # Level 1
        d1 = self.up1(x5)           # Upsample bottleneck: 512 ch
        # Apply DA conditionally
        skip4 = self.da1(x4) if self.da_level == 1 else x4
        d1 = torch.cat([skip4, d1], dim=1) # Concatenate skip4 + d1 = 1024 ch
        d1 = self.conv1(d1)         # Process combined -> 512 ch

        # Level 2
        d2 = self.up2(d1)           # Upsample d1: 256 ch
        # Apply DA conditionally
        skip3 = self.da2(x3) if self.da_level == 2 else x3
        d2 = torch.cat([skip3, d2], dim=1) # Concatenate skip3 + d2 = 512 ch
        d2 = self.conv2(d2)         # Process combined -> 256 ch
        d2_att = self.cbam_dec2(d2) # Apply CBAM after decoder conv2

        # Level 3
        d3 = self.up3(d2_att)       # Upsample CBAM'd d2: 128 ch
        # Apply DA conditionally
        skip2 = self.da3(x2) if self.da_level == 3 else x2
        d3 = torch.cat([skip2, d3], dim=1) # Concatenate skip2 + d3 = 256 ch
        d3 = self.conv3(d3)         # Process combined -> 128 ch

        # Level 4
        d4 = self.up4(d3)           # Upsample d3: 64 ch
        # Apply DA conditionally
        skip1 = self.da4(x1) if self.da_level == 4 else x1
        d4 = torch.cat([skip1, d4], dim=1) # Concatenate skip1 + d4 = 128 ch
        d4 = self.conv4(d4)         # Process combined -> 64 ch
        d4_att = self.cbam_dec4(d4) # Apply CBAM after decoder conv4

        # Final output
        logits = self.outc(d4_att)  # Output from CBAM'd d4
        return logits