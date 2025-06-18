"""TensorFlow DropBlock implementation."""

import tensorflow as tf

class DropBlock2D(tf.keras.layers.Layer):
    """DropBlock regularization for 2D feature maps."""

    def __init__(self, block_size=7, keep_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob

    def call(self, inputs, training=None):
        if training is False or self.keep_prob == 1.0:
            return inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training:
            return inputs

        input_shape = tf.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]

        gamma = ((1.0 - self.keep_prob) * tf.cast(height * width, tf.float32) /
                 (self.block_size ** 2) /
                 tf.cast((height - self.block_size + 1) * (width - self.block_size + 1), tf.float32))

        bernoulli = tf.cast(tf.random.uniform(input_shape, dtype=tf.float32) < gamma, tf.float32)
        mask = -tf.nn.max_pool2d(-bernoulli, ksize=self.block_size, strides=1, padding='SAME')
        mask = 1.0 - mask

        return inputs * mask * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
