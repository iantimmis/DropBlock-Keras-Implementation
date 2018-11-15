from keras import backend as K
from keras.engine.topology import Layer
from scipy.stats import bernoulli
import copy

class DropBlock(Layer):
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
    def __init__(self, block_size, keep_prob, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob

    def call(self, x, training=None):

        # During inference, we do not Drop Blocks. (Similar to DropOut)
        if training == None:
            return x

        # Calculate Gamma
        feat_size = int(x.shape[-1])
        gamma = ((1-self.keep_prob)/(self.block_size**2)) * ((feat_size**2) / ((feat_size-self.block_size+1)**2))

        padding = self.block_size//2

        # Randomly sample mask
        sample = bernoulli.rvs(size=(feat_size-(padding*2), feat_size-(padding*2)),p=gamma)

        # The above code creates a matrix of zeros and samples ones from the distribution
        # We would like to flip all of these values
        sample = 1-sample

        # Pad the mask with ones
        sample = np.pad(sample, pad_width=padding, mode='constant', constant_values=1)

        # For each 0, create spatial square mask of shape (block_size x block_size)
        mask = copy.copy(sample)
        for i in range(feat_size):
            for j in range(feat_size):
                if sample[i, j]==0:
                    mask[i-padding : i+padding+1, j-padding : j+padding+1] = 0

        mask = mask.reshape((1, feat_size, feat_size))

        # Apply the mask
        x = x * np.repeat(mask, x.shape[1], 0)

        # Normalize the features
        count = np.prod(mask.shape)
        count_ones = np.count_nonzero(mask == 1)
        x = x * count / count_ones

        return x

    def get_config(self):
        config = {'block_size': self.block_size,
                  'gamma': self.gamma,
                  'seed': self.seed}
        base_config = super(DropBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
