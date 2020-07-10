# DropBlock-Keras-Implementation
Paper Reproduction: "DropBlock: A regularization method for convolutional networks" (Google Brain, https://arxiv.org/abs/1810.12890)

# Requirements
```bash
pip install keras numpy scipy
```

# DropBlock
DropBlock was designed to regularize Convolutional Neural Networks. According to the paper,

```
(a) input image to a convolutional neural network. The green regions in (b) and (c) include
the activation units which contain semantic information in the input image. Dropping out activations
at random is not effective in removing semantic information because nearby activations contain
closely related information. Instead, dropping continuous regions can remove certain semantic
information (e.g., head or feet) and consequently enforcing remaining units to learn features for
classifying input image."
```

![intuition](https://github.com/iantimmis/DropBlock-Keras-Implementation/blob/master/images/Intuition.png)

The blocks are selected as follows,

```
Mask sampling in DropBlock. (a) On every feature map, similar to dropout, we first
sample a mask M. We only sample mask from shaded green region in which each sampled entry can
expanded to a mask fully contained inside the feature map. (b) Every zero entry on M is expanded to
block_size × block_size zero block.
```

![drop_block](https://github.com/iantimmis/DropBlock-Keras-Implementation/blob/master/images/DropBlock.png)

# Usage
```python
from DropBlock import DropBlock

DropBlock(block_size=5, keep_prob=.9)
```
