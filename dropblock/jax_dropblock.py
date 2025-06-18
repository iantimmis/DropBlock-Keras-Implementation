"""JAX DropBlock implementation."""

from typing import Optional
import jax
import jax.numpy as jnp


def dropblock2d(x: jnp.ndarray, block_size: int = 7, keep_prob: float = 0.9, training: bool = True, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
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
    # During inference, we do not Drop Blocks. (Similar to DropOut)
    if not training or keep_prob == 1.0:
        return x
    if rng is None:
        rng = jax.random.PRNGKey(0)

    h, w = x.shape[1], x.shape[2]
    # Calculate Gamma
    gamma = (1.0 - keep_prob) * h * w / (block_size ** 2) / ((h - block_size + 1) * (w - block_size + 1))
    # Randomly sample mask
    bernoulli = jax.random.uniform(rng, x.shape) < gamma
    bernoulli = bernoulli.astype(jnp.float32)
    # For each 0, create spatial square mask of shape (block_size x block_size)
    mask = -jax.lax.reduce_window(-bernoulli, 0.0, jax.lax.max, (1, block_size, block_size, 1), (1, 1, 1, 1), "SAME")
    mask = 1.0 - mask
    # Apply the mask and normalize the features
    return x * mask * mask.size / jnp.sum(mask)
