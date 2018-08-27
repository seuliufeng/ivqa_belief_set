import numpy as np
import tensorflow as tf


def gather_by_col(arr, cols):
    batch_size = tf.shape(arr)[0]
    num_cols = tf.shape(arr)[1]
    index = tf.range(batch_size) * num_cols + cols
    arr = tf.reshape(arr, [-1])
    return tf.reshape(tf.gather(arr, index), [batch_size])


def test():
    arr = np.random.rand(6, 5)
    position = np.random.randint(low=0, high=5, size=(6,))
    arr_t = tf.constant(arr)
    position_t = tf.constant(position, dtype=tf.int32)
    col_t = gather_by_col(arr_t, position_t)
    sess = tf.Session()
    print('Arr:')
    print(sess.run(arr_t))
    print('Cols:')
    print(sess.run(position_t))
    print('Vals:')
    print(sess.run(col_t))


if __name__ == '__main__':
    test()

