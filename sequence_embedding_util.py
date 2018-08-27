import tensorflow as tf

sess = tf.InteractiveSession()


def convert_to_multihot_coding(labels, num_classes):
    embedding = tf.eye(num_classes, num_classes,
                       dtype=tf.float32)
    embedding = tf.concat([embedding,
                           tf.zeros([1, num_classes], dtype=tf.float32)],
                          axis=0)  # num_classes+1 x num_classes
    mask = tf.cast(labels < num_classes, tf.float32)  # NxT
    labels = tf.minimum(labels, num_classes)
    embeded = tf.nn.embedding_lookup(embedding, labels)  # NxTxD
    mh_coding = tf.reduce_sum(tf.expand_dims(mask, 2) * embeded,
                              axis=1)
    mh_coding = tf.cast(mh_coding > 0., dtype=tf.float32)
    return mh_coding


def averge_pool_sequence(embeded, mask, EPS=1e-8):
    """
    Average pooling along the temporal dimension
    :param embeded: embeded features, NxTxD
    :param mask: input mask, NxT
    :return:
    """
    mask = tf.expand_dims(mask, 2)  # NxTx1
    coding = tf.reduce_sum(mask * embeded, axis=1)  # NxD
    valid_counts = tf.reduce_sum(mask, 1)  # Nx1
    avg_pooled = tf.divide(coding, valid_counts + EPS)
    return avg_pooled


class Word2VecSquenceEncoder(object):
    def __init__(self, vocab_size,
                 w2v_coding='data/vqa_trainval_answer_vocab_w2v.data'):
        self._vocab_size = vocab_size
        from util import load_hdf5
        w2v = load_hdf5(w2v_coding)['ans_w2v']
        w2v_t = tf.Variable(initial_value=w2v,
                            trainable=False,
                            dtype=tf.float32)
        dummy = tf.zeros([1, 300], dtype=tf.float32)
        self._w2v = tf.concat([w2v_t, dummy], axis=0)

    def encode(self, tokens):
        mask = tf.cast(tokens < self._vocab_size, tf.float32)  # NxT
        tokens = tf.minimum(tokens, self._vocab_size)
        embeded = tf.nn.embedding_lookup(self._w2v, tokens)  # NxTxD
        coding = averge_pool_sequence(embeded, mask)
        return tf.nn.l2_normalize(coding, dim=1)


def test_multihot_coding():
    import numpy as np
    batch_size = 500
    vocab_size = 26
    seq_len = 300
    coding_len = 25
    # x = np.random.rand(batch_size, vocab_size).argsort(axis=1)[:, :seq_len]
    x = np.random.randint(low=0, high=vocab_size,
                          size=(batch_size, seq_len),
                          dtype=np.int32)
    tx = tf.constant(x, dtype=tf.int32)
    coding = convert_to_multihot_coding(tx, coding_len)
    mhc = coding.eval()
    assert (mhc.max() == 1.)
    # assert (mhc.min() == 0.)
    assert (np.sum(mhc == 1.) + np.sum(mhc == 0.) == mhc.size)
    print('Value check: Passed')
    for x_i, c in zip(x, mhc):
        nnz = np.where(c > 0)[0]
        gt = x_i[x_i < coding_len]
        assert (np.intersect1d(nnz, gt).size == nnz.size)
    print('Index check: Passed')


if __name__ == '__main__':
    test_multihot_coding()
