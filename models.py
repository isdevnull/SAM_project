__all__ = ["ResNet"]

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_f: int, out_f: int, stride: int, bias: bool = False):
        super().__init__()
        self.in_f, self.out_f, self.stride = in_f, out_f, stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_f),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_f),
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.in_f != self.out_f or self.stride != 1:
            pad = self.out_f // 4
            padded_x = F.pad(x[..., ::2, ::2], (0, 0, 0, 0, pad, pad))
            out += padded_x
        else:
            out += x
        return F.relu(out)


class ResNet(torch.nn.Module):
    def __init__(
        self,
        ch: int,
        n_blocks: List[int],
        strides: List[int],
        n_classes: int,
        bias: bool = False,
    ):
        super().__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=3, padding=1, bias=bias), nn.BatchNorm2d(ch)
        )
        start_ch = ch
        self.main_blocks = nn.ModuleList()
        for block_size, stride in zip(n_blocks, strides):
            for i in range(block_size):
                if i == 0:
                    cur_stride = stride
                else:
                    cur_stride = 1
                self.main_blocks.append(
                    ResNetBlock(start_ch, ch, stride=cur_stride, bias=bias)
                )
                start_ch = ch
            ch *= 2

        self.fc = nn.Linear(start_ch, n_classes)

    def forward(self, x):
        x = self.start_conv(x)
        for layer in self.main_blocks:
            x = layer(x)
        x = F.avg_pool2d(x, x.shape[-1])
        return self.fc(x.view(x.shape[0], x.shape[1]))
