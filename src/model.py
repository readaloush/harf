"""HarfNet: a small residual CNN for 32x32 grayscale letter images."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + (x if self.skip is None else self.skip(x)))


class HarfNet(nn.Module):
    def __init__(self, num_classes: int = 28, width: int = 32, dropout: float = 0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.stages = nn.Sequential(
            Block(w, w),              # 32x32
            Block(w, 2 * w, 2),       # 16x16
            Block(2 * w, 2 * w),
            Block(2 * w, 4 * w, 2),   # 8x8
            Block(4 * w, 4 * w),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
