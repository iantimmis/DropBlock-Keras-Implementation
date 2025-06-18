"""TensorFlow DropBlock implementation."""



import tensorflow as tf

class DropBlock2D(tf.keras.layers.Layer):
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

    def __init__(self, block_size=7, keep_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob

    def call(self, inputs, training=None):
        # During inference, we do not Drop Blocks. (Similar to DropOut)
        if training is False or self.keep_prob == 1.0:
            return inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training:
            return inputs

        input_shape = tf.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]

        # Calculate Gamma
        gamma = ((1.0 - self.keep_prob) * tf.cast(height * width, tf.float32) /
                 (self.block_size ** 2) /
                 tf.cast((height - self.block_size + 1) * (width - self.block_size + 1), tf.float32))

        # Randomly sample mask
        bernoulli = tf.cast(tf.random.uniform(input_shape, dtype=tf.float32) < gamma, tf.float32)
        # For each 0, create spatial square mask of shape (block_size x block_size)
        mask = -tf.nn.max_pool2d(-bernoulli, ksize=self.block_size, strides=1, padding='SAME')
        mask = 1.0 - mask

        # Apply the mask and normalize the features
        return inputs * mask * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
