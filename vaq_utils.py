import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from ops import softmax_attention, mlb
from attention_ops import semantic_attention
from rnn_ops import _create_drop_lstm_cell, _build_caption_inputs_and_targets
from greedy_decoding import build_nic_greedy_decoder, build_attention_greedy_decoder
from attention_cells import ConditionalSoftAttentionCell, ConditionalSoftAttentionCell2, \
    ConditionalSoftAttentionCell3, ConditionalSoftAttentionCell4, ConditionalSoftAttentionCell5, \
    BaselineSoftAttentionCell
from attention_beamsearch_decoder import build_vaq_dec_attention_predictor
from ops import concat_op
from rnn_compact_ops import *

_START_TOKEN_ID = 1
_END_TOKEN_ID = 2


def build_attribute_guided_vaq_model(im, attribute, ans_embed, quest, quest_len, embed_dim,
                                     vocab_size, keep_prob, pad_token, num_dec_cells,
                                     phase='train'):
    # average pooling over image
    fv = softmax_attention(im, attribute, embed_dim, keep_prob=keep_prob,
                           scope='AttributeAttention')
    in_embed = tf.concat(1, [fv, attribute, ans_embed])
    with tf.variable_scope('vaq'):
        if phase == 'train':
            inputs, targets, length = _build_caption_inputs_and_targets(quest,
                                                                        quest_len)
            return build_lstm_decoder(in_embed, inputs, length, targets,
                                      vocab_size, num_dec_cells, keep_prob,
                                      pad_token)
        else:
            return build_lstm_predictor(in_embed, quest, vocab_size,
                                        num_dec_cells, pad_token)


def build_vaq_model(im, ans_embed, quest, quest_len, vocab_size,
                    keep_prob, pad_token, num_dec_cells):
    inputs, targets, length = _build_caption_inputs_and_targets(quest, quest_len)
    # average pooling over image
    im = tf.reduce_mean(im, reduction_indices=[1, 2])
    in_embed = tf.concat(concat_dim=1, values=[im, ans_embed])
    with tf.variable_scope('vaq'):
        return build_lstm_decoder(in_embed, inputs, length, targets,
                                  vocab_size, num_dec_cells, keep_prob, pad_token)


def build_attention_vaq_model(im, ans_embed, quest, quest_len, embed_dim, vocab_size,
                              keep_prob, pad_token, num_dec_cells, phase='train'):
    # average pooling over image
    fv = softmax_attention(im, ans_embed, embed_dim, keep_prob=keep_prob,
                           scope='AnsAttention')
    in_embed = mlb(fv, ans_embed, embed_dim, keep_prob, scope='VAEmbed')
    with tf.variable_scope('vaq'):
        if phase == 'train':
            inputs, targets, length = _build_caption_inputs_and_targets(quest,
                                                                        quest_len)
            return build_lstm_decoder(in_embed, inputs, length, targets,
                                      vocab_size, num_dec_cells, keep_prob,
                                      pad_token)
        else:
            return build_lstm_predictor(in_embed, quest, vocab_size,
                                        num_dec_cells, pad_token)


def build_vanilla_vaq_model(im, ans_embed, quest, quest_len, embed_dim, vocab_size,
                            keep_prob, pad_token, num_dec_cells, phase='train'):
    # average pooling over image
    in_embed = tf.concat(1, values=[im, ans_embed])
    with tf.variable_scope('vaq'):
        if phase == 'train' or phase == 'condition':
            inputs, targets, length = _build_caption_inputs_and_targets(quest,
                                                                        quest_len)
            return build_lstm_decoder(in_embed, inputs, length, targets,
                                      vocab_size, num_dec_cells, keep_prob,
                                      pad_token)
        elif phase == 'greedy':
            return build_nic_greedy_decoder(in_embed, vocab_size, num_dec_cells,
                                            _START_TOKEN_ID)
        else:
            return build_lstm_predictor(in_embed, quest, vocab_size,
                                        num_dec_cells, pad_token)


def build_decoding_attention_vaq_model(im, attr, ans_embed, quest, quest_len, vocab_size,
                                       keep_prob, pad_token, num_dec_cells, phase='train',
                                       cell_option=1):
    if attr is None:
        in_embed = ans_embed
    else:
        in_embed = concat_op(values=[attr, ans_embed], axis=1)

    with tf.variable_scope('vaq'):
        if phase == 'train' or phase == 'condition' or phase == 'evaluate':
            inputs, targets, length = _build_caption_inputs_and_targets(quest,
                                                                        quest_len)
            return build_attention_decoder(in_embed, im, ans_embed, inputs, length, targets,
                                           vocab_size, num_dec_cells, keep_prob,
                                           pad_token, cell_option)
        elif phase == 'greedy':
            return build_attention_greedy_decoder(in_embed, im, ans_embed,
                                                  vocab_size, num_dec_cells,
                                                  pad_token, cell_option)
        else:
            return build_vaq_dec_attention_predictor(in_embed, im, ans_embed,
                                                  quest, vocab_size, num_dec_cells,
                                                  pad_token, cell_option)


def build_att2_vaq_model(im, ans_embed, quest, quest_len, embed_dim, vocab_size,
                         keep_prob, pad_token, num_dec_cells, phase='train'):
    # average pooling over image
    fv = semantic_attention(im, ans_embed, embed_dim, keep_prob)
    # in_embed = mlb(fv, ans_embed, embed_dim, keep_prob, scope='in_embed')
    in_embed = tf.concat(concat_dim=1, values=[fv, ans_embed])
    with tf.variable_scope('vaq'):
        if phase == 'train':
            inputs, targets, length = _build_caption_inputs_and_targets(quest,
                                                                        quest_len)
            return build_lstm_decoder(in_embed, inputs, length, targets,
                                      vocab_size, num_dec_cells, keep_prob,
                                      pad_token)
        else:
            return build_lstm_predictor(in_embed, quest, vocab_size,
                                        num_dec_cells, pad_token)


def build_lstm_decoder(in_embed, inputs, length, targets, vocab_size,
                       num_cells, keep_prob, pad_token):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(in_embed, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    inputs = tf.nn.embedding_lookup(word_map, inputs)

    # build LSTM cell and RNN
    lstm = _create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                  output_keep_prob=keep_prob,
                                  cell_fn=tf.nn.rnn_cell.BasicLSTMCell)
    outputs, states = tf.nn.dynamic_rnn(lstm, inputs, length, initial_state=init_state,
                                        dtype=tf.float32)

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    # compute loss
    batch_size = tf.shape(targets)[0]
    targets = tf.reshape(targets, [-1])
    mask = tf.cast(tf.not_equal(targets, pad_token), tf.float32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
                  name='dec_loss')
    slim.losses.add_loss(loss)
    return tf.reshape(losses * mask, [batch_size, -1])


def build_attention_decoder(glb_ctx, im, ans, inputs, length, targets, vocab_size,
                            num_cells, keep_prob, pad_token):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state_lstm = LSTMStateTuple(init_c, init_h)
    batch_size = tf.shape(init_h)[0]
    att_zero_state = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = (init_state_lstm, att_zero_state)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    inputs = tf.nn.embedding_lookup(word_map, inputs)

    # build LSTM cell and RNN
    lstm = _create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                  output_keep_prob=keep_prob,
                                  cell_fn=BasicLSTMCell)

    cell_keep_prob = 1.0

    attention_cell = ConditionalSoftAttentionCell4(512, im, ans, keep_prob=cell_keep_prob)

    attention_cell = DropoutWrapper(attention_cell, input_keep_prob=1.0,
                                                   output_keep_prob=keep_prob)

    multi_cell = MultiRNNCell([lstm, attention_cell], state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(multi_cell, inputs, length, initial_state=init_state,
                                   dtype=tf.float32)

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    # compute loss
    batch_size = tf.shape(targets)[0]
    targets = tf.reshape(targets, [-1])
    mask = tf.cast(tf.not_equal(targets, pad_token), tf.float32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
                  name='dec_loss')
    slim.losses.add_loss(loss)
    return tf.reshape(losses * mask, [batch_size, -1])
    # return tf.reshape(losses * mask, [batch_size, -1]), attmaps


def build_lstm_predictor(in_embed, inputs, vocab_size, num_cells, pad_token):
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)

    # init state / image embedding
    init_h = slim.fully_connected(in_embed, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h)

    # build LSTM cell and RNN
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_cells)
    tf.concat(1, init_state, name="initial_state")

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
    feed_c, feed_h = tf.split(1, 2, state_feed)
    state_tuple = tf.nn.rnn_cell.LSTMStateTuple(feed_c, feed_h)

    # Run a single LSTM step.
    with tf.variable_scope('RNN'):
        outputs, state_tuple = lstm_cell(inputs=tf.squeeze(word_embed,
                                                           squeeze_dims=[1]),
                                         state=state_tuple)

    # Concatentate the resulting state.
    state = tf.concat(1, state_tuple, name="state")

    # Stack batches vertically.
    outputs = tf.reshape(outputs, [-1, lstm_cell.output_size])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    prob = tf.nn.softmax(logits, name="softmax")
    return state


def process_input_data(inputs, pad_token):
    im, capt, capt_len, a = inputs
    capt_len = capt_len.flatten()
    pad = np.ones(shape=[capt_len.size, 1], dtype=np.int32)
    capt = np.concatenate((_START_TOKEN_ID * pad, capt, pad), axis=1)
    capt_len += 2
    for x, x_len in zip(capt, capt_len):
        x[x_len - 1] = _END_TOKEN_ID
        x[x_len:] = pad_token
    return [im, capt, capt_len, a]


if __name__ == '__main__':
    from scipy.io import loadmat

    d = loadmat('debug_capt_model.mat')
    q = d['q']
    q_len = d['q_len']
    im = d['im']
    a = d['a']
    print(q[:2])
    _, q, q_len, _ = process_input_data([im, q, q_len, a], pad_token=-1)
    print(q[:2])
