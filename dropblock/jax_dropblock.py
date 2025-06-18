"""JAX DropBlock implementation."""

from typing import Optional
import jax
import jax.numpy as jnp


def dropblock2d(x: jnp.ndarray, block_size: int = 7, keep_prob: float = 0.9, training: bool = True, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Applies DropBlock to a NHWC array."""
    if not training or keep_prob == 1.0:
        return x
    if rng is None:
        rng = jax.random.PRNGKey(0)

    h, w = x.shape[1], x.shape[2]
    gamma = (1.0 - keep_prob) * h * w / (block_size ** 2) / ((h - block_size + 1) * (w - block_size + 1))
    bernoulli = jax.random.uniform(rng, x.shape) < gamma
    bernoulli = bernoulli.astype(jnp.float32)
    mask = -jax.lax.reduce_window(-bernoulli, 0.0, jax.lax.max, (1, block_size, block_size, 1), (1, 1, 1, 1), "SAME")
    mask = 1.0 - mask
    return x * mask * mask.size / jnp.sum(mask)
