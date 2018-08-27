import tensorflow as tf
import tensorflow.contrib.slim as slim
from rnn_ops import *
from beam_search_util.tf_beam_decoder import beam_decoder
import vaq_utils


def _build_greedy_inference_decoder(in_embed, vocab_size, num_cells, start_token_id):
    vocab_size += 1
    # init state / image embedding
    init_h = slim.fully_connected(in_embed, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = LSTMStateTuple(init_c, init_h)

    # build LSTM cell and RNN
    lstm_cell = BasicLSTMCell(num_cells)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))

    # apply weights for outputs
    with tf.variable_scope('logits'):
        weights = tf.get_variable('weights', shape=[num_cells, vocab_size], dtype=tf.float32)
        biases = tf.get_variable('biases', shape=[vocab_size])

    # define helper functions
    def _tokens_to_inputs_fn(inputs):
        inputs = tf.nn.embedding_lookup(word_map, inputs)
        inputs = tf.squeeze(inputs, [1])
        return inputs

    def _output_to_score_fn(hidden):
        return tf.nn.xw_plus_b(hidden, weights, biases)

    stop_token_id = vaq_utils._END_TOKEN_ID
    batch_size = tf.shape(in_embed, 0)
    start_tokens = tf.ones(shape=[batch_size], dtype=tf.int32) * start_token_id
    init_inputs = _tokens_to_inputs_fn(start_tokens)
    cand_symbols, cand_logprobs = beam_decoder(lstm_cell, beam_size=3,
                                               stop_token=stop_token_id,
                                               initial_state=init_state,
                                               initial_input=init_inputs,
                                               tokens_to_inputs_fn=_tokens_to_inputs_fn,
                                               outputs_to_score_fn=_output_to_score_fn,
                                               max_len=20,
                                               output_dense=True,
                                               scope='RNN')
    return cand_symbols, cand_logprobs
