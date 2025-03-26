# modules_1sobel_v3/unet_ch6.py (수정)
import torch
import torch.nn as nn
from modules_1sobel_v3.double_conv import DoubleConv  # Import DoubleConv
from modules_1sobel_v3.attention import CBAM  # Import CBAM

# Helper function for normalization (you can adjust this)
def norm(dim, type="batch"):
    if type == "batch":
        return nn.BatchNorm2d(dim)
    elif type == "group":
        return nn.GroupNorm(32, dim) # 32 is a common group size
    else:
        return nn.Identity()

# --- Copy DANetHead and related modules here ---
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling parameter, initialized to 0

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (H*W) X (H*W)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X (H*W) X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (H*W)
        energy = torch.bmm(proj_query, proj_key) # Transpose check, batch matrix multiplication: B X (H*W) X (H*W)
        attention = self.softmax(energy) #  B X (H*W) X (H*W)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # B X C X (H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B X C X (H*W)
        out = out.view(m_batchsize, C, height, width)  # B X C X H X W

        out = self.gamma*out + x  # Scaled addition with the original input
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling parameter
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1) # B X C X (H*W)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B X (H*W) X C
        energy = torch.bmm(proj_query, proj_key) # B X C X C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)  # B X C X (H*W)
        out = out.view(m_batchsize, C, height, width) # B X C X H X W

        out = self.gamma*out + x
        return out

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 16  # Intermediate channel dimension
        # inter_channels = in_channels

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)  # Position Attention Module
        self.sc = CAM_Module(inter_channels)  # Channel Attention Module

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)
        return sasc_output


class UNet(nn.Module):
    def __init__(self, n_channels=7, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.cbam64 = CBAM(64)
        self.cbam128 = CBAM(128)
        self.cbam256 = CBAM(256)
        self.cbam512 = CBAM(512)
        self.cbam1024 = CBAM(1024)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 2048))

        # Decoder with DANetHeads on skip connections
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.da1 = DANetHead(1024, 1024)  # DA for skip connection
        self.up1_conv = DoubleConv(2048, 1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.da2 = DANetHead(512, 512)
        self.up2_conv = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.da3 = DANetHead(256, 256)
        self.up3_conv = DoubleConv(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.da4 = DANetHead(128, 128)
        self.up4_conv = DoubleConv(256, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.da5 = DANetHead(64, 64)
        self.up5_conv = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.cbam64(x1)
        x3 = self.down1(x2)
        x3 = self.cbam128(x3)
        x4 = self.down2(x3)
        x4 = self.cbam256(x4)
        x5 = self.down3(x4)
        x5 = self.cbam512(x5)
        x6 = self.down4(x5)
        x6 = self.cbam1024(x6)
        x7 = self.down5(x6)


        x = self.up1(x7)
        x = self.up1_conv(torch.cat([self.da1(x6), x], dim=1))  # Apply DA to skip connection

        x = self.up2(x)
        x = self.up2_conv(torch.cat([self.da2(x5), x], dim=1))

        x = self.up3(x)
        x = self.up3_conv(torch.cat([self.da3(x4), x], dim=1))

        x = self.up4(x)
        x = self.up4_conv(torch.cat([self.da4(x3), x], dim=1))

        x = self.up5(x)
        x = self.up5_conv(torch.cat([self.da5(x2), x], dim=1))

        x = self.outc(x)
        return x