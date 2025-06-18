"""DropBlock regularization modules for multiple frameworks."""

from .tf_dropblock import DropBlock2D as TFDropBlock2D  # noqa: F401
from .torch_dropblock import DropBlock2D as TorchDropBlock2D  # noqa: F401
from .jax_dropblock import dropblock2d as jax_dropblock2d  # noqa: F401
