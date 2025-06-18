"""PyTorch DropBlock implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D feature maps."""

    def __init__(self, block_size=7, keep_prob=0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1.0:
            return x
        gamma = self._compute_gamma(x)
        bernoulli = (torch.rand_like(x) < gamma).float()
        mask = -F.max_pool2d(-bernoulli, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1.0 - mask
        return x * mask * mask.numel() / mask.sum()

    def _compute_gamma(self, x):
        h, w = x.shape[2], x.shape[3]
        return ((1.0 - self.keep_prob) * h * w / (self.block_size ** 2) /
                ((h - self.block_size + 1) * (w - self.block_size + 1)))
