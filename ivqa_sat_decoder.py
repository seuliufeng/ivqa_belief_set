import tensorflow as tf
import tensorflow.contrib.slim as slim
from rnn_ops import create_drop_lstm_cell
from rnn_ops import build_caption_inputs_and_targets
from greedy_decoding import create_greedy_decoder
from ops import concat_op, split_op
from rnn_compact_ops import *
from ivqa_rnn_cells import ShowAttendTellCell
from beam_search_util.tf_beam_decoder import beam_decoder
from config import VOCAB_CONFIG


def build_decoder(im, attr, ans_embed, quest, quest_len, vocab_size,
                  keep_prob, pad_token, num_dec_cells, phase='train',
                  add_loss=True):
    if attr is None:
        in_embed = ans_embed
    else:
        in_embed = concat_op(values=[attr, ans_embed], axis=1)

    with tf.variable_scope('inverse_vqa'):
        if phase == 'train' or phase == 'condition' or phase == 'evaluate':
            inputs, targets, length = build_caption_inputs_and_targets(quest,
                                                                       quest_len)
            return _build_training_decoder(in_embed, im, ans_embed, inputs, length, targets,
                                           vocab_size, num_dec_cells, keep_prob,
                                           pad_token, add_loss)
        elif phase == 'greedy':
            return _build_greedy_inference_decoder(in_embed, im, ans_embed,
                                                   vocab_size, num_dec_cells,
                                                   VOCAB_CONFIG.start_token_id)
        elif phase == 'beam':
            return _build_tf_beam_inference_decoder(in_embed, im, ans_embed,
                                                    vocab_size, num_dec_cells,
                                                    VOCAB_CONFIG.start_token_id,
                                                    pad_token)
        else:
            return _build_beamsearch_inference_decoder(in_embed, im, ans_embed,
                                                       quest, vocab_size, num_dec_cells,
                                                       pad_token)


# *****************  TRAINING GRAPH *******************************
def _build_training_decoder(glb_ctx, im, ans, inputs, length, targets, vocab_size,
                            num_cells, keep_prob, pad_token, add_loss=True):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_h = slim.dropout(init_h, keep_prob=keep_prob)
    init_c = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_c')
    init_c = slim.dropout(init_c, keep_prob=keep_prob)
    init_state = LSTMStateTuple(init_c, init_h)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    inputs = tf.nn.embedding_lookup(word_map, inputs)
    # inputs = slim.dropout(inputs, keep_prob=keep_prob)

    # build LSTM cell and RNN
    cell = ShowAttendTellCell(512, im, keep_prob=keep_prob)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, length, initial_state=init_state,
                                   dtype=tf.float32, scope='RNN')
    outputs = slim.dropout(outputs, keep_prob=keep_prob)

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

    if add_loss:
        loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
                      name='dec_loss')
        slim.losses.add_loss(loss)
        return tf.reshape(losses * mask, [batch_size, -1])
    else:
        mask = tf.reshape(mask, [batch_size, -1])
        losses = tf.reshape(losses, [batch_size, -1]) * mask
        return losses, mask


# *****************  Greedy Decoding GRAPH ***************************
def _build_greedy_inference_decoder(glb_ctx, im, ans_embed, vocab_size,
                                    num_cells, start_token_id):
    vocab_size += 1
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_lstm_state = LSTMStateTuple(init_c, init_h)
    batch_size = tf.shape(init_h)[0]
    att_zero_state = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = (init_lstm_state, att_zero_state)

    # build LSTM cell and RNN
    lstm_cell = BasicLSTMCell(num_cells)
    attention_cell = MultiModalAttentionCell(512, im, ans_embed, keep_prob=1.0)
    multi_cell = MultiRNNCell([lstm_cell, attention_cell], state_is_tuple=True)

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

    return create_greedy_decoder(init_state, multi_cell, word_map,
                                 softmax_params, start_token_id)


# *****************  Beam Search GRAPH *******************************
def _build_tf_beam_inference_decoder(glb_ctx, im, ans_embed, vocab_size,
                                     num_cells, start_token_id, pad_token):
    beam_size = 3
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_c')
    init_state = concat_op([init_c, init_h], axis=1)

    # replicate context of the attention module
    im_shape = im.get_shape().as_list()[1:]
    im = tf.expand_dims(im, 1)  # add a time dim
    im = tf.reshape(tf.tile(im, [1, beam_size, 1, 1, 1]), [-1] + im_shape)

    multi_cell = ShowAttendTellCell(num_cells, im, state_is_tuple=False)

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

    stop_token_id = VOCAB_CONFIG.end_token_id
    batch_size = tf.shape(glb_ctx)[0]
    start_tokens = tf.ones(shape=[batch_size], dtype=tf.int32) * start_token_id
    init_inputs = _tokens_to_inputs_fn(start_tokens)
    pathes, scores = beam_decoder(multi_cell, beam_size=beam_size,
                                  stop_token=stop_token_id,
                                  initial_state=init_state,
                                  initial_input=init_inputs,
                                  tokens_to_inputs_fn=_tokens_to_inputs_fn,
                                  outputs_to_score_fn=_output_to_score_fn,
                                  max_len=20,
                                  cell_transform='flatten',
                                  output_dense=True,
                                  scope='RNN')
    return scores, pathes
