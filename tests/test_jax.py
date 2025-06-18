import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from dropblock.jax_dropblock import dropblock2d


def test_jax_dropblock_shape():
    x = jnp.ones((1, 8, 8, 3))
    y = dropblock2d(x, block_size=3, keep_prob=0.9, training=True)
    assert y.shape == x.shape
