"""PyTorch DropBlock implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    """
    Regularization Technique for Convolutional Layers.

    Pseudocode:
    1: Input:output activations of a layer (A), block_size, γ, mode
    2: if mode == Inference then
    3: return A
    4: end if
    5: Randomly sample mask M: Mi,j ∼ Bernoulli(γ)
    6: For each zero position Mi,j , create a spatial square mask with the center being Mi,j , the width,
        height being block_size and set all the values of M in the square to be zero (see Figure 2).
    7: Apply the mask: A = A × M
    8: Normalize the features: A = A × count(M)/count_ones(M)

    # Arguments
        block_size: A Python integer. The size of the block to be dropped.
        gamma: float between 0 and 1. controls how many activation units to drop.
    # References
        - [DropBlock: A regularization method for convolutional networks](
           https://arxiv.org/pdf/1810.12890v1.pdf)
    """

    def __init__(self, block_size=7, keep_prob=0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        # During inference, we do not Drop Blocks. (Similar to DropOut)
        if not self.training or self.keep_prob == 1.0:
            return x
        # Calculate Gamma
        gamma = self._compute_gamma(x)
        # Randomly sample mask
        bernoulli = (torch.rand_like(x) < gamma).float()
        # For each 0, create spatial square mask of shape (block_size x block_size)
        mask = -F.max_pool2d(-bernoulli, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1.0 - mask
        # Apply the mask and normalize the features
        return x * mask * mask.numel() / mask.sum()

    def _compute_gamma(self, x):
        h, w = x.shape[2], x.shape[3]
        return ((1.0 - self.keep_prob) * h * w / (self.block_size ** 2) /
                ((h - self.block_size + 1) * (w - self.block_size + 1)))
