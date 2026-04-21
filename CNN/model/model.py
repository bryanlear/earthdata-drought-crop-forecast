"""
CIFAR-style ResNet-18 with masked region pooling for multi-region drought classification.
Steps: 3×3 stem (no stride/maxpool) → 4 ResNet layers (64→512 ch, 64→8 spatial)
       → masked average pool per region → shared FC head → (B, R, 3) logits.
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

    def __init__(self, in_channels: int = 9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)
        self.out_channels = 512

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


class MaskedRegionPool(nn.Module):

    def forward(self,
                feat: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        _, R, _, _ = masks.shape


        m = masks.float()
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode='nearest')


        m_exp    = m.unsqueeze(2)
        feat_exp = feat.unsqueeze(1)

        numerator   = (feat_exp * m_exp).sum(dim=(-2, -1))
        denominator = m_exp.sum(dim=(-2, -1)).clamp(min=1.0)
        return numerator / denominator


class DroughtCNN(nn.Module):

    def __init__(self,
                 in_channels: int = 9,
                 n_classes:   int = 3,
                 dropout:     float = 0.3):
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels)
        self.pool     = MaskedRegionPool()
        self.head     = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self,
                image: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        feat   = self.backbone(image)
        pooled = self.pool(feat, masks)
        B, R, C = pooled.shape
        logits = self.head(pooled.view(B * R, C))
        return logits.view(B, R, -1)
