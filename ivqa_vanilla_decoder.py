import tensorflow as tf
from ops import concat_op, split_op
import tensorflow.contrib.slim as slim
from rnn_ops import create_drop_lstm_cell
from rnn_ops import build_caption_inputs_and_targets
from greedy_decoding import create_greedy_decoder
from random_decoding import create_greedy_decoder as create_random_decoder
from rnn_compact_ops import *
from beam_search_util.tf_beam_decoder import beam_decoder

_START_TOKEN_ID = 1
_END_TOKEN_ID = 2


def build_decoder(im, ans_embed, quest, quest_len, vocab_size,
                  keep_prob, pad_token, num_dec_cells, phase='train'):
    # average pooling over image
    in_embed = concat_op(values=[im, ans_embed], axis=1)
    with tf.variable_scope('vaq'):
        if phase == 'train' or phase == 'condition':
            inputs, targets, length = build_caption_inputs_and_targets(quest,
                                                                       quest_len)
            return _build_training_decoder(in_embed, inputs, length, targets,
                                           vocab_size, num_dec_cells, keep_prob,
                                           pad_token)
        elif phase == 'greedy':
            return _build_greedy_inference_decoder(in_embed, vocab_size, num_dec_cells,
                                                   _START_TOKEN_ID)
        elif phase == 'sampling':
            return _build_random_inference_decoder(in_embed, vocab_size, num_dec_cells,
                                                   _START_TOKEN_ID)
        elif phase == 'beam':
            return _build_tf_beam_inference_decoder(in_embed, vocab_size, num_dec_cells,
                                                    _START_TOKEN_ID)
        else:
            return _build_beamsearch_inference_decoder(in_embed, quest, vocab_size,
                                                       num_dec_cells, pad_token)


# *****************  TRAINING GRAPH *******************************
def _build_training_decoder(in_embed, inputs, length, targets, vocab_size,
                            num_cells, keep_prob, pad_token):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(in_embed, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = LSTMStateTuple(init_c, init_h)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    inputs = tf.nn.embedding_lookup(word_map, inputs)

    # build LSTM cell and RNN
    lstm = create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                 output_keep_prob=keep_prob,
                                 cell_fn=BasicLSTMCell)
    outputs, states = tf.nn.dynamic_rnn(lstm, inputs, length, initial_state=init_state,
                                        dtype=tf.float32, scope='RNN')

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    # compute loss
    batch_size = tf.shape(targets)[0]
    targets = tf.reshape(targets, [-1])
    mask = tf.cast(tf.not_equal(targets, pad_token), tf.float32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=targets)
    loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
                  name='dec_loss')
    slim.losses.add_loss(loss)
    return tf.reshape(losses * mask, [batch_size, -1])


# *****************  Greedy Decoding GRAPH ***************************
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
        softmax_params = [weights, biases]

    return create_greedy_decoder(init_state, lstm_cell, word_map,
                                 softmax_params, start_token_id)


# *****************  Random Decoding GRAPH ***************************
def _build_random_inference_decoder(in_embed, vocab_size, num_cells, start_token_id):
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
        softmax_params = [weights, biases]

    return create_random_decoder(init_state, lstm_cell, word_map,
                                 softmax_params, start_token_id, epsion=1.0)


# *****************  Beam Search GRAPH *******************************
def _build_beamsearch_inference_decoder(in_embed, inputs, vocab_size, num_cells, pad_token):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)

    # init state / image embedding
    init_h = slim.fully_connected(in_embed, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = LSTMStateTuple(init_c, init_h)

    # build LSTM cell and RNN
    lstm_cell = BasicLSTMCell(num_cells)
    concat_op(init_state, axis=1, name="initial_state")

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    word_embed = tf.nn.embedding_lookup(word_map, inputs)

    # Placeholder for feeding a batch of concatenated states.
    state_feed = tf.placeholder(dtype=tf.float32,
                                shape=[None, sum(lstm_cell.state_size)],
                                name="state_feed")
    feed_c, feed_h = split_op(state_feed, num_splits=2, axis=1)
    state_tuple = LSTMStateTuple(feed_c, feed_h)

    # Run a single LSTM step.
    with tf.variable_scope('RNN'):
        outputs, state_tuple = lstm_cell(inputs=tf.squeeze(word_embed,
                                                           squeeze_dims=[1]),
                                         state=state_tuple)

    # Concatentate the resulting state.
    state = concat_op(state_tuple, 1, name="state")

    # Stack batches vertically.
    outputs = tf.reshape(outputs, [-1, lstm_cell.output_size])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    prob = tf.nn.softmax(logits, name="softmax")
    return state


# *****************  Beam Search GRAPH *******************************
def _build_tf_beam_inference_decoder(in_embed, vocab_size, num_cells, start_token_id):
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
        # inputs = tf.squeeze(inputs, [1])
        return inputs

    def _output_to_score_fn(hidden):
        batch_size = tf.shape(hidden)[0]
        beam_size = tf.shape(hidden)[1]
        hidden = tf.reshape(hidden, [batch_size * beam_size, -1])
        logits = tf.nn.xw_plus_b(hidden, weights, biases)
        logprob = tf.nn.log_softmax(logits)
        return tf.reshape(logprob, [batch_size, beam_size, -1])

    stop_token_id = _END_TOKEN_ID
    batch_size = tf.shape(in_embed)[0]
    start_tokens = tf.ones(shape=[batch_size], dtype=tf.int32) * start_token_id
    init_inputs = _tokens_to_inputs_fn(start_tokens)
    pathes, scores = beam_decoder(lstm_cell, beam_size=3,
                                  stop_token=stop_token_id,
                                  initial_state=init_state,
                                  initial_input=init_inputs,
                                  tokens_to_inputs_fn=_tokens_to_inputs_fn,
                                  outputs_to_score_fn=_output_to_score_fn,
                                  max_len=20,
                                  output_dense=True,
                                  scope='RNN')
    return scores, pathes
