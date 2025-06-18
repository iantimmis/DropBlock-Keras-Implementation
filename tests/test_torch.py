import pytest

torch = pytest.importorskip("torch")
from dropblock.torch_dropblock import DropBlock2D


def test_torch_dropblock_shape():
    layer = DropBlock2D(block_size=3, keep_prob=0.9)
    layer.train()
    x = torch.ones((1, 3, 8, 8))
    y = layer(x)
    assert y.shape == x.shape
