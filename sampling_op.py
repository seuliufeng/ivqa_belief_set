import tensorflow as tf


def deterministic_sampling(logits):
    index = tf.argmax(logits, dimension=1)
    return tf.to_int32(index)


def stochastic_sampling(logits):
    index = tf.multinomial(logits, num_samples=1)
    return tf.to_int32(tf.reshape(index, [-1]))

