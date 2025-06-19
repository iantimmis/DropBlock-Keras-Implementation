<div align="center">
<h1> DropBlock Unofficial Implementation </h1>

This project contains an **unofficial** set of implementations of
["DropBlock: A regularization method for convolutional networks"](https://arxiv.org/abs/1810.12890)
from Google Brain. DropBlock is a variant of dropout that removes contiguous
regions from feature maps instead of individual activations.
These layers can be used to regularize convolutional networks across multiple
machine learning frameworks.

</div>

## Requirements
DropBlock itself only depends on NumPy. You will also need at least one of the
supported deep learning frameworks if you want to run the examples or tests.

* Python 3.8 or later
* NumPy
* Optional: [TensorFlow](https://www.tensorflow.org/),
  [PyTorch](https://pytorch.org/) or [JAX](https://github.com/google/jax)

Install the packages you require, e.g.:
```bash
pip install numpy tensorflow torch jax
```

## About
The paper proposes dropping spatial blocks of activations during training so
that nearby units cannot simply co-adapt. In practice a mask is sampled with
a probability `gamma` and expanded into square regions of zeros. This has been
shown to improve generalization on several vision benchmarks.

This repository is provided for educational purposes and is **not** an official
release from the authors.

![intuition](https://github.com/iantimmis/DropBlock-Keras-Implementation/blob/master/images/Intuition.png)

## Quick start
Each framework has its own API under the `dropblock` package.

### PyTorch
```python
from dropblock.torch_dropblock import DropBlock2D
layer = DropBlock2D(block_size=5, keep_prob=0.9)
```

### TensorFlow / Keras
```python
from dropblock.tf_dropblock import DropBlock2D
layer = DropBlock2D(block_size=5, keep_prob=0.9)
```

### JAX
```python
from dropblock.jax_dropblock import dropblock2d
output = dropblock2d(x, block_size=5, keep_prob=0.9, training=True)
```

## Testing
Run unit tests (they automatically skip if the corresponding framework is not installed):
```bash
pytest -q
```
