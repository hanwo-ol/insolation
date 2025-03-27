# modules_1sobel_DA/attention_da.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint # Import checkpoint

# Choose normalization layer (BatchNorm or GroupNorm)
# norm_layer = nn.BatchNorm2d
def norm_layer(channels):
    # Use GroupNorm if batch size might be small, otherwise BatchNorm
    # Adjust num_groups as needed, 32 is a common value
    num_groups = max(1, channels // 8) # Ensure num_groups is at least 1
    # Ensure channels is divisible by num_groups if possible, or handle edge case
    if channels % num_groups != 0:
        # Find the largest divisor of channels <= channels // 8, or default to 1 or 4
        if channels >= 32 and channels % 8 == 0: num_groups = channels // 8
        elif channels >= 16 and channels % 4 == 0: num_groups = channels // 4
        elif channels % 2 == 0: num_groups = 2
        else: num_groups = 1 # Fallback for prime or small channel numbers
        # print(f"Adjusted num_groups to {num_groups} for {channels} channels") # Optional: for debugging

    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    # return nn.BatchNorm2d(channels)


class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from https://github.com/junfu1115/DANet

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # Use 1x1 convolutions for query, key, value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling factor

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key) # Batch matrix multiplication
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x # Add residual connection
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    # Ref from https://github.com/junfu1115/DANet

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling factor
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) # Batch matrix multiplication
        # Original CAM uses this normalization:
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # Alternative (simpler softmax on energy - might also work):
        # attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x # Add residual connection
        return out

class DualAttentionModule(nn.Module):
    """
    Dual Attention Module adapted for skip connections, optimized for memory.
    Applies Position Attention and Channel Attention.
    """
    def __init__(self, in_channels, reduction_factor=16, use_checkpoint=True):
        super(DualAttentionModule, self).__init__()
        if in_channels < reduction_factor : # Ensure inter_channels is at least 1
             inter_channels = max(1, in_channels // 4) # Use a smaller reduction if needed
        else:
            inter_channels = in_channels // reduction_factor

        if inter_channels == 0:
            inter_channels = 1 # Avoid zero channels

        self.use_checkpoint = use_checkpoint

        self.conv_in_pa = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_in_ca = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.pam = PAM_Module(inter_channels)
        self.cam = CAM_Module(inter_channels)

        self.conv_out_pa = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out_ca = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Output convolutions to restore original channel dimension
        # Use dropout as in the reference
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.05, False),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1) # Output same channels as input
            # Consider adding a final Norm + ReLU if needed by subsequent layers
            # norm_layer(in_channels),
            # nn.ReLU(inplace=True)
        )

    # Define separate functions for checkpointing
    def forward_pam_wrapper(self, feat_pa):
        return self.pam(feat_pa)

    def forward_cam_wrapper(self, feat_ca):
        return self.cam(feat_ca)


    def forward(self, x):
        # Position Attention Branch
        feat_pa = self.conv_in_pa(x)
        if self.use_checkpoint and self.training:
            # Pass the wrapper function to checkpoint
            sa_feat = checkpoint(self.forward_pam_wrapper, feat_pa, use_reentrant=False) # Use use_reentrant=False for newer PyTorch versions
        else:
            sa_feat = self.pam(feat_pa) # Direct call if not checkpointing or not training
        sa_conv = self.conv_out_pa(sa_feat)

        # Channel Attention Branch
        feat_ca = self.conv_in_ca(x)
        if self.use_checkpoint and self.training:
            # Pass the wrapper function to checkpoint
            sc_feat = checkpoint(self.forward_cam_wrapper, feat_ca, use_reentrant=False) # Use use_reentrant=False
        else:
            sc_feat = self.cam(feat_ca) # Direct call
        sc_conv = self.conv_out_ca(sc_feat)

        # Combine features (element-wise sum as in DANet paper)
        feat_sum = sa_conv + sc_conv

        # Final output convolution
        output = self.conv_out(feat_sum)

        return output
