import tensorflow as tf
from util import get_default_initializer
from rnn_compact_ops import *


class LSTMEncoder(object):
    def __init__(self, vocab_size=None, word_embedding=None,
                 keep_prob=1.0, word_embed_dim=300, num_cells=256,
                 scope='LSTMEnc'):
        assert (vocab_size is not None or word_embedding is not None)
        self._vocab_size = vocab_size
        self._scope = scope
        self._word_embed = word_embedding
        self._word_embed_dim = word_embed_dim
        self._num_cells = num_cells
        self._keep_prob = keep_prob
        lstm1 = BasicLSTMCell(num_units=num_cells, state_is_tuple=True)
        lstm1 = DropoutWrapper(lstm1, input_keep_prob=self._keep_prob,
                               output_keep_prob=self._keep_prob)
        self._cells = lstm1

    def __call__(self, inputs, inputs_len, initial_state=None):
        with tf.variable_scope(self._scope) as sc:
            if self._word_embed is None:
                self._word_embed = tf.get_variable(
                    name="map",
                    shape=[self._vocab_size, self._word_embed_dim],
                    initializer=get_default_initializer())

            print('WordEmbed')
            print(self._word_embed)
            seq_embeds = tf.nn.embedding_lookup(self._word_embed, inputs)
            _, states = tf.nn.dynamic_rnn(self._cells, seq_embeds, inputs_len,
                                          initial_state=initial_state,
                                          dtype=tf.float32, scope=sc)
            final_state = tf.concat(states, axis=1)
        return final_state


if __name__ == '__main__':
    lstm = LSTMEncoder(vocab_size=100)
    import numpy as np

    seq = np.random.randint(low=0, high=100, size=[3, 4])
    seq_len = np.random.randint(low=1, high=4, size=(3,))
    seq_t = tf.constant(seq, dtype=tf.int32)
    seq_len_t = tf.constant(seq_len, dtype=tf.int32)
    outputs = lstm(seq_t, seq_len_t)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    o = sess.run(outputs)
    print(o.shape)
