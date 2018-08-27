import numpy as np
import tensorflow as tf


def test_boolean_mask():
    N = 6
    x = np.random.rand(N, 5)
    print('x:')
    print(x)

    ind = np.random.randint(low=0, high=5, size=(N,))
    print('ind:')
    print(ind)

    x_tf = tf.constant(x, dtype=tf.float32)
    mask_tf = tf.constant(ind, dtype=tf.int32)
    values = _gather_by_index(x_tf, mask_tf)

    sess = tf.Session()
    m = sess.run(values)
    print('res:')
    print(m)


def _gather_by_index(scores, inds):
    scores_flat = tf.reshape(scores, [-1])
    batch_size = tf.shape(scores)[0]
    vocab_size = tf.shape(scores)[1]
    _indices = tf.range(batch_size) * vocab_size + inds
    values = tf.gather(scores_flat, _indices)
    return values



if __name__ == '__main__':
    test_boolean_mask()
