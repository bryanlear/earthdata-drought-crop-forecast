"""
CIFAR-style ResNet-18 for single-region drought classification.

Input:  (B, 10, 64, 64) — 8 SMAP + 1 CHIRPS + 1 region mask channel
Month:  (B,) int        — month-of-year 1..12 (optional)
Output: (B, n_classes)  — drought class logits

Architecture:
  - CIFAR ResNet-18 backbone (3×3 stem, no stride, no maxpool)
    width multiplier scales channels (0.25 ≈ 700K params)
  - AdaptiveAvgPool2d → flatten
  - Optional concat of (sin, cos) month encoding
  - FC head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18Backbone(nn.Module):

    def __init__(self, in_channels: int = 10, width: float = 1.0):
        super().__init__()

        def w(n: int) -> int:
            return max(1, int(n * width))

        c1, c2, c3, c4 = w(64), w(128), w(256), w(512)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(c1, c1, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(c1, c2, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(c2, c3, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(c3, c4, n_blocks=2, stride=2)
        self.out_channels = c4

    @staticmethod
    def _make_layer(in_planes: int, planes: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DroughtCNN(nn.Module):
    """Single-region drought classifier with optional month-of-year encoding."""

    def __init__(self,
                 in_channels: int = 10,
                 n_classes: int = 2,
                 dropout: float = 0.3,
                 width: float = 1.0,
                 use_month: bool = True):
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels, width=width)
        self.use_month = use_month
        feat_dim = self.backbone.out_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(32, feat_dim // 4)
        head_in_dim = feat_dim + (2 if use_month else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self,
                image: torch.Tensor,
                month: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(image)
        pooled = self.gap(feat).flatten(1)

        if self.use_month:
            if month is None:
                raise ValueError('month tensor is required when use_month=True')
            angle = 2.0 * torch.pi * month.float() / 12.0
            month_enc = torch.stack([angle.sin(), angle.cos()], dim=1)
            pooled = torch.cat([pooled, month_enc], dim=1)

        return self.head(pooled)