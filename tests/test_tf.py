import pytest

tf = pytest.importorskip("tensorflow")
from dropblock.tf_dropblock import DropBlock2D


def test_tf_dropblock_shape():
    layer = DropBlock2D(block_size=3, keep_prob=0.9)
    x = tf.ones((1, 8, 8, 3))
    y = layer(x, training=True)
    assert y.shape == x.shape
