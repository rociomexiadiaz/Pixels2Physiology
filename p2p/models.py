"""The two neural networks and their losses.

UNet          - stage 1, finds the paper in the photo (1 input channel).
UNet_Res_CBAM - stage 3, finds the ink trace on each strip (2 input channels:
                grayscale + FFT magnitude).
Class names match the training notebook so the released .pth files load as-is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, chs=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.down1 = DoubleConv(1, chs[0])
        self.down2 = DoubleConv(chs[0], chs[1])
        self.down3 = DoubleConv(chs[1], chs[2])
        self.down4 = DoubleConv(chs[2], chs[3])
        self.down5 = DoubleConv(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.conv4 = DoubleConv(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.conv3 = DoubleConv(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.conv2 = DoubleConv(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.conv1 = DoubleConv(chs[1], chs[0])
        self.out = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bottleneck = self.down5(self.pool(d4))

        x = self.conv4(torch.cat([self.up4(bottleneck), d4], dim=1))
        x = self.conv3(torch.cat([self.up3(x), d3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), d2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), d1], dim=1))
        return self.out(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + res)


class CBAM(nn.Module):
    """Channel attention then spatial attention (Woo et al., 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * channel_att
        avg_ch = torch.mean(x, dim=1, keepdim=True)
        max_ch, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_ch, max_ch], dim=1))


class UNet_Res_CBAM(nn.Module):
    def __init__(self, chs=(64, 128, 256, 512, 1024), in_ch=2):
        super().__init__()
        self.down1 = ResBlock(in_ch, chs[0])
        self.down2 = ResBlock(chs[0], chs[1])
        self.down3 = ResBlock(chs[1], chs[2])
        self.down4 = ResBlock(chs[2], chs[3])
        self.down5 = ResBlock(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.cbam4 = CBAM(chs[3])
        self.cbam3 = CBAM(chs[2])
        self.cbam2 = CBAM(chs[1])
        self.cbam1 = CBAM(chs[0])

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.conv4 = ResBlock(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.conv3 = ResBlock(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.conv2 = ResBlock(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.conv1 = ResBlock(chs[1], chs[0])
        self.out = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bottleneck = self.down5(self.pool(d4))

        x = self.conv4(torch.cat([self.up4(bottleneck), self.cbam4(d4)], 1))
        x = self.conv3(torch.cat([self.up3(x), self.cbam3(d3)], 1))
        x = self.conv2(torch.cat([self.up2(x), self.cbam2(d2)], 1))
        x = self.conv1(torch.cat([self.up1(x), self.cbam1(d1)], 1))
        return self.out(x)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum(dim=(2, 3))
        den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return (1 - (num + self.eps) / (den + self.eps)).mean()


class BCEDiceLoss(nn.Module):
    """UNet-1 objective: 0.5 BCE + 0.5 Dice."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


class DiceFocalLoss(nn.Module):
    """UNet-2 objective. Note: the 'focal' term is plain mean BCE, as trained."""

    def __init__(self, dice_weight=0.5, focal_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        inter = (p * target).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()
        focal = F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean()
        return self.dice_weight * dice + self.focal_weight * focal
