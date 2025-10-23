import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .decouplennet import DecoupleNet_D2_1662_e64_k9_drop01

# ---------------------- RGB -> HSV ----------------------
class RGB_to_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_to_HSV, self).__init__()
        self.eps = eps

    def forward(self, im):
        img = im * 0.5 + 0.5  # [-1,1] -> [0,1]
        maxc, _ = img.max(dim=1)
        minc, _ = img.min(dim=1)
        delta = maxc - minc + self.eps

        hue = torch.zeros_like(maxc)
        idx = (img[:, 2] == maxc)
        hue[idx] = 4.0 + (img[:, 0][idx] - img[:, 1][idx]) / delta[idx]
        idx = (img[:, 1] == maxc)
        hue[idx] = 2.0 + (img[:, 2][idx] - img[:, 0][idx]) / delta[idx]
        idx = (img[:, 0] == maxc)
        hue[idx] = ((img[:, 1][idx] - img[:, 2][idx]) / delta[idx]) % 6.0
        hue[minc == maxc] = 0.0
        hue = hue / 6.0

        saturation = delta / (maxc + self.eps)
        saturation[maxc == 0] = 0
        value = maxc

        hsv = torch.stack([hue, saturation, value], dim=1)
        return hsv * 2 - 1  # [0,1] -> [-1,1]

# ---------------------- RGB -> Lab ----------------------
class RGB_to_Lab(nn.Module):
    def __init__(self):
        super(RGB_to_Lab, self).__init__()
        self.rgb_to_xyz = torch.tensor([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ]).float()

    def forward(self, x):
        x = x * 0.5 + 0.5
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz = torch.matmul(x, self.rgb_to_xyz.to(x.device))
        xyz = xyz.view(B, H, W, 3).permute(0, 3, 1, 2)
        xyz_ref = torch.tensor([0.9505, 1.0000, 1.0890]).view(1, 3, 1, 1).to(x.device)
        xyz = xyz / xyz_ref
        delta = 6.0 / 29.0
        mask = xyz > delta**3
        lab = torch.zeros_like(xyz)
        lab[:, 0] = 116.0 * torch.where(mask[:, 1], xyz[:, 1]**(1/3), (xyz[:, 1] / (3 * delta**2)) + 4/29) - 16
        lab[:, 1] = 500.0 * (torch.where(mask[:, 0], xyz[:, 0]**(1/3), (xyz[:, 0] / (3 * delta**2)) + 4/29) -
                             torch.where(mask[:, 1], xyz[:, 1]**(1/3), (xyz[:, 1] / (3 * delta**2)) + 4/29))
        lab[:, 2] = 200.0 * (torch.where(mask[:, 1], xyz[:, 1]**(1/3), (xyz[:, 1] / (3 * delta**2)) + 4/29) -
                             torch.where(mask[:, 2], xyz[:, 2]**(1/3), (xyz[:, 2] / (3 * delta**2)) + 4/29))
        lab[:, 0] = lab[:, 0] / 50.0 - 1.0
        lab[:, 1:] = lab[:, 1:] / 128.0
        return lab

# ---------------------- conv_block ----------------------
def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# ---------------------- Channel Attention ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ---------------------- PixelAttention ----------------------
class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

# ---------------------- 颜色空间注意力融合 ----------------------
class ColorSpaceAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ColorSpaceAttentionFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, hsv_feat, lab_feat):
        fusion = torch.cat([rgb_feat, hsv_feat, lab_feat], dim=1)
        weights = self.fc(self.global_pool(fusion))  # [B, 3, 1, 1]
        rgb_weight = weights[:, 0:1, :, :]
        hsv_weight = weights[:, 1:2, :, :]
        lab_weight = weights[:, 2:3, :, :]
        out = rgb_feat * rgb_weight + hsv_feat * hsv_weight + lab_feat * lab_weight
        return out

# ---------------------- Feature Fusion Module (FFM) 修改后的----------------------
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureFusionModule, self).__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 通道注意力
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.sigmoid_c = nn.Sigmoid()

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 最终融合
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)               # [B, C1+C2, H, W]
        x = self.fusion(x)                           # [B, Cout, H, W]

        ### 通道注意力分支
        ch_input = x.clone()
        ch_att = self.global_pool(x)
        ch_att = self.fc1(ch_att)
        ch_att = self.relu(ch_att)
        ch_att = self.fc2(ch_att)
        ch_att = self.sigmoid_c(ch_att)
        x_ch = x * ch_att + ch_input                 # 残差连接

        ### 空间注意力分支
        sp_input = x.clone()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = torch.cat([avg_out, max_out], dim=1)
        sp_att = self.spatial_att(sp_att)
        x_sp = x * sp_att + sp_input                 # 残差连接

        ### 合并两条路径（推荐：拼接后1×1卷积压缩）
        x_out = torch.cat([x_ch, x_sp], dim=1)       # [B, Cout*2, H, W]
        out = self.out_conv(x_out)                   # [B, Cout, H, W]

        return out


# ---------------------- 空间特征提取器 ----------------------
class SpatialFeatureExtractor(nn.Module):
    def __init__(self):
        super(SpatialFeatureExtractor, self).__init__()
        self.rgb2hsv = RGB_to_HSV()
        self.rgb2lab = RGB_to_Lab()

        self.RGB_initial_convolution_block = conv_block(3, 64, stride=1)
        self.HSV_initial_convolution_block = conv_block(3, 64, stride=1)
        self.Lab_initial_convolution_block = conv_block(3, 64, stride=1)

        self.down2 = conv_block(64, 128, stride=2)
        self.down3 = conv_block(64, 128, stride=2)
        self.down7 = conv_block(64, 128, stride=2)

        self.down5 = conv_block(128, 256, stride=2)
        self.down6 = conv_block(128, 256, stride=2)
        self.down8 = conv_block(128, 256, stride=2)

        self.down9 = conv_block(256, 512, stride=2)
        self.down10 = conv_block(256, 512, stride=2)
        self.down11 = conv_block(256, 512, stride=2)

        self.fusion1 = ColorSpaceAttentionFusion(64)
        self.fusion2 = ColorSpaceAttentionFusion(128)
        self.fusion3 = ColorSpaceAttentionFusion(256)
        self.fusion4 = ColorSpaceAttentionFusion(512)

        self.ca_s1 = ChannelAttention(64)
        self.pa_s1 = PixelAttention(64)
        self.ca_s2 = ChannelAttention(128)
        self.pa_s2 = PixelAttention(128)
        self.ca_s4 = ChannelAttention(256)
        self.pa_s4 = PixelAttention(256)
        self.ca_s8 = ChannelAttention(512)
        self.pa_s8 = PixelAttention(512)

    def forward(self, x):
        HSV1 = self.rgb2hsv(x)
        Lab1 = self.rgb2lab(x)

        HSV2 = self.HSV_initial_convolution_block(HSV1)
        Lab2 = self.Lab_initial_convolution_block(Lab1)
        RGB1 = self.RGB_initial_convolution_block(x)
        RGB2 = self.fusion1(RGB1, HSV2, Lab2)
        RGB2 = self.ca_s1(RGB2)
        RGB2 = self.pa_s1(RGB2)

        HSV3 = self.down2(HSV2)
        Lab3 = self.down3(Lab2)
        RGB3 = self.down7(RGB2)
        RGB4 = self.fusion2(RGB3, HSV3, Lab3)
        RGB4 = self.ca_s2(RGB4)
        RGB4 = self.pa_s2(RGB4)

        HSV4 = self.down5(HSV3)
        Lab4 = self.down6(Lab3)
        RGB5 = self.down8(RGB4)
        RGB6 = self.fusion3(RGB5, HSV4, Lab4)
        RGB6 = self.ca_s4(RGB6)
        RGB6 = self.pa_s4(RGB6)

        HSV7 = self.down9(HSV4)
        Lab7 = self.down10(Lab4)
        RGB7 = self.down11(RGB6)
        RGB8 = self.fusion4(RGB7, HSV7, Lab7)
        RGB8 = self.ca_s8(RGB8)
        RGB8 = self.pa_s8(RGB8)

        return RGB2, RGB4, RGB6, RGB8

# ---------------------- 语义特征提取器 ----------------------
class SemanticFeatureExtractor(nn.Module):
    def __init__(self):
        super(SemanticFeatureExtractor, self).__init__()
        self.decouple_net = DecoupleNet_D2_1662_e64_k9_drop01(fork_feat=True)

        # 替换为多层非线性 MLP 模块（1x1 conv + ReLU + 1x1 conv）
        def mlp_block(in_channels, out_channels, mid_channels=None):
            mid_channels = mid_channels or out_channels
            return nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.mlp_s1 = mlp_block(64, 64, 128)
        self.mlp_s2 = mlp_block(128, 128, 256)
        self.mlp_s4 = mlp_block(256, 256, 512)
        self.mlp_s8 = mlp_block(512, 512, 512)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.decouple_net(x)

        # print('0',features[0].shape)
        # print('1',features[1].shape)
        # print('2',features[2].shape)
        # print('3',features[3].shape)

        x1 = F.interpolate(features[0], size=(H, W), mode='bilinear', align_corners=True)
        x1 = self.mlp_s1(x1)  # → [B, 64, H, W]
        # print('x1', x1.shape)

        x2 = F.interpolate(features[1], size=(H//2, W//2), mode='bilinear', align_corners=True)
        x2 = self.mlp_s2(x2)  # → [B, 128, H/2, W/2]
        # print('x2', x2.shape)

        x3 = F.interpolate(features[2], size=(H//4, W//4), mode='bilinear', align_corners=True)
        x3 = self.mlp_s4(x3)  # → [B, 256, H/4, W/4]
        # print('x3', x3.shape)

        x4 = F.interpolate(features[3], size=(H//8, W//8), mode='bilinear', align_corners=True)
        x4 = self.mlp_s8(x4)  # → [B, 512, H/8, W/8]
        # print('x4', x4.shape)

        return x1, x2, x3, x4



# ---------------------- Shadow Detection Network ----------------------
class ShadowDetectionNetwork(nn.Module):
    def __init__(self):
        super(ShadowDetectionNetwork, self).__init__()
        self.spatial_extractor = SpatialFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor()

        self.ffm_c1 = FeatureFusionModule(64, 64, 64)
        self.ffm_c2 = FeatureFusionModule(128, 128, 128)
        self.ffm_c3 = FeatureFusionModule(256, 256, 256)
        self.ffm_c4 = FeatureFusionModule(512, 512, 512)

        self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_c6 = conv_block(512, 256, stride=1)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_c8 = conv_block(256, 128, stride=1)
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_c10 = conv_block(128, 64, stride=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        RGB2, RGB4, RGB6, RGB8 = self.spatial_extractor(x)
        x1, x2, x3, x4 = self.semantic_extractor(x)

        c1 = self.ffm_c1(RGB2, x1)
        c2 = self.ffm_c2(RGB4, x2)
        c3 = self.ffm_c3(RGB6, x3)
        c4 = self.ffm_c4(RGB8, x4)

        c5 = self.up_conv1(c4)
        c5 = torch.cat([c5, c3], dim=1)
        c6 = self.conv_c6(c5)

        c7 = self.up_conv2(c6)
        c7 = torch.cat([c7, c2], dim=1)
        c8 = self.conv_c8(c7)

        c9 = self.up_conv3(c8)
        c9 = torch.cat([c9, c1], dim=1)
        c10 = self.conv_c10(c9)

        out = self.final_conv(c10)
        out = torch.sigmoid(out)
        return out

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShadowDetectionNetwork().to(device)
    x = torch.randn(6, 3, 512, 512).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print("\n=== 特征提取器输出尺寸 ===")
    print(f"out: {out.shape}")
