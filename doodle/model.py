"""Небольшая CNN для рисунков 64×64: 4 блока (2× conv3×3-BN-ReLU + maxpool) → GAP → dropout → FC.

Только операции, которые есть в Unity Inference Engine: Conv, BatchNormalization (при экспорте
вливается в Conv), Relu, MaxPool, GlobalAveragePool, Flatten, Gemm, Softmax.
Ширины (32, 64, 128, 256) ≈ 1.2 M параметров.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_bn(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoodleNet(nn.Module):
    def __init__(self, num_classes: int, widths=(32, 64, 128, 256), dropout: float = 0.2):
        super().__init__()
        layers = []
        cin = 1
        for i, w in enumerate(widths):
            layers += [conv_bn(cin, w), conv_bn(w, w)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        self.config = {"num_classes": num_classes, "widths": list(widths), "dropout": dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, N, N), 0 = фон, 1 = чернила → логиты (B, C)."""
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(self.drop(x))


class ExportWrapper(nn.Module):
    """Для ONNX: логиты / T → softmax. T — температура калибровки (scripts/evaluate.py)."""

    def __init__(self, net: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.net = net
        self.inv_t = float(1.0 / temperature)

    def forward(self, x):
        z = self.net(x)
        if self.inv_t != 1.0:
            z = z * self.inv_t
        return torch.softmax(z, dim=1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fuse_for_export(net: DoodleNet) -> DoodleNet:
    """Влить BatchNorm в Conv (eval-режим) — меньше операций в графе, тот же результат."""
    import copy

    from torch.nn.utils.fusion import fuse_conv_bn_eval

    net = copy.deepcopy(net).eval()
    for i, blk in enumerate(net.features):
        if isinstance(blk, nn.Sequential) and isinstance(blk[0], nn.Conv2d):
            fused = fuse_conv_bn_eval(blk[0], blk[1])
            net.features[i] = nn.Sequential(fused, nn.ReLU(inplace=False))
    return net


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DoodleNet(**ck["model_config"])
    net.load_state_dict(ck["model"])
    net.to(device).eval()
    return net, ck
