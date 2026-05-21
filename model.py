"""
model.py
========
CNN + ASPP classifier for GPM precipitation classification from AGRI patches.

Architecture:
  Input: (B, C+2, 33, 33) — 7 BT channels + 2 geo (lat, lon)

  Block1: Conv3×3(9→64) + BN + ReLU                   → (B, 64, 33, 33)
  Block2: ResBlock(64→128, stride=2)                   → (B, 128, 17, 17)
  Block3: ResBlock(128→256, stride=2)                  → (B, 256, 9, 9)
  Block4: ResBlock(256→256, dil=2)                     → (B, 256, 9, 9)
  Block5: ResBlock(256→256, dil=4)                     → (B, 256, 9, 9)

  ASPP (multi-scale context):
    1×1 + dil6 + dil12 + global pool → concat → 1×1    → (B, 256, 9, 9)

  GlobalAvgPool → Dropout(0.3) → Linear(256→128) → Dropout(0.2) → Linear(128→4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ResBlock(nn.Module):
    """Residual block with optional stride and dilation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = out_ch // 4

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_d6 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_d12 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        b1 = self.branch_1x1(x)
        b2 = self.branch_d6(x)
        b3 = self.branch_d12(x)
        b4 = self.branch_pool(x)
        b4 = F.interpolate(b4, size=(h, w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))


class PrecipClassifier(nn.Module):
    """CNN + ASPP classifier for precipitation patches.

    Input:  (B, AGRI_CHANNELS+GEO_CHANNELS, H, W)  e.g. (B, 9, 33, 33)
    Output: (B, NUM_CLASSES)  class logits
    """

    def __init__(
        self,
        in_channels: int = cfg.AGRI_CHANNELS + cfg.GEO_CHANNELS,
        num_classes: int = cfg.NUM_CLASSES,
    ):
        super().__init__()

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── ResBlocks ──
        self.block1 = ResBlock(64, 128, stride=2, dilation=1)    # 33→17
        self.block2 = ResBlock(128, 256, stride=2, dilation=1)   # 17→9
        self.block3 = ResBlock(256, 256, stride=1, dilation=2)   # 9→9
        self.block4 = ResBlock(256, 256, stride=1, dilation=4)   # 9→9

        # ── ASPP ──
        self.aspp = ASPP(256, 256)

        # ── Head ──
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)         # (B, 64, 33, 33)
        x = self.block1(x)       # (B, 128, 17, 17)
        x = self.block2(x)       # (B, 256, 9, 9)
        x = self.block3(x)       # (B, 256, 9, 9)
        x = self.block4(x)       # (B, 256, 9, 9)
        x = self.aspp(x)         # (B, 256, 9, 9)
        x = self.pool(x)         # (B, 256, 1, 1)
        x = x.flatten(1)         # (B, 256)
        x = self.drop1(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop2(x)
        return self.fc2(x)       # (B, 4)


def build_model() -> PrecipClassifier:
    return PrecipClassifier(
        in_channels=cfg.AGRI_CHANNELS + cfg.GEO_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    )


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, cfg.AGRI_CHANNELS + cfg.GEO_CHANNELS, cfg.PATCH_SIZE[0], cfg.PATCH_SIZE[1])
    out = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {total:,}")
