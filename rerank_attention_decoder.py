import tensorflow as tf
import tensorflow.contrib.slim as slim
from rnn_ops import create_drop_lstm_cell
from rnn_ops import build_caption_inputs_and_targets
from greedy_decoding import create_greedy_decoder
from ops import concat_op, split_op
from rnn_compact_ops import *
from ivqa_rnn_cells import MultiModalAttentionCell, RerankAttentionCell
from beam_search_util.tf_beam_decoder import beam_decoder
from config import VOCAB_CONFIG


def build_decoder(im, attr, ans_embed, quest, quest_len, vocab_size,
                  keep_prob, pad_token, num_dec_cells, phase='train'):
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
                                           pad_token)
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
    lstm = create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                 output_keep_prob=keep_prob,
                                 cell_fn=BasicLSTMCell)

    attention_cell = RerankAttentionCell(512, im, ans, keep_prob=keep_prob)
    attention_cell = DropoutWrapper(attention_cell, input_keep_prob=1.0,
                                    output_keep_prob=keep_prob)

    multi_cell = MultiRNNCell([lstm, attention_cell], state_is_tuple=True)
    outputs, final_states = tf.nn.dynamic_rnn(multi_cell, inputs, length,
                                              initial_state=init_state,
                                              dtype=tf.float32, scope='RNN')
    final_att_state = final_states[1]
    rerank_logits = slim.fully_connected(final_att_state, 1, activation_fn=None,
                                         scope='Softmax')
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

    mask = tf.reshape(mask, [batch_size, -1])
    losses = tf.reshape(losses, [batch_size, -1]) * mask

    # loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
    #               name='dec_loss')
    # slim.losses.add_loss(loss)
    # return rerank_logits
    return rerank_logits, losses, mask


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
def _build_beamsearch_inference_decoder(glb_ctx, im, ans, inputs, vocab_size,
                                        num_cells, pad_token):
    vocab_size = max(vocab_size, pad_token + 1)

    # =================== create cells ========================
    lstm = BasicLSTMCell(num_cells)
    attention_cell = MultiModalAttentionCell(512, im, ans, keep_prob=1.0)

    multi_cell = MultiRNNCell([lstm, attention_cell],
                              state_is_tuple=True)
    lstm_state_sizes, att_state_size = multi_cell.state_size
    lstm_state_size = sum(lstm_state_sizes)
    state_size = lstm_state_size + att_state_size

    # =============  create state placeholders ================
    state_feed = tf.placeholder(dtype=tf.float32, shape=[None, state_size],
                                name='state_feed')
    lstm_state_feed = tf.slice(state_feed, begin=[0, 0],
                               size=[-1, lstm_state_size])
    att_state_feed = tf.slice(state_feed, begin=[0, lstm_state_size],
                              size=[-1, att_state_size])
    feed_c, feed_h = split_op(lstm_state_feed, num_splits=2, axis=1)
    state_tuple = LSTMStateTuple(feed_c, feed_h)
    multi_cell_state_feed = (state_tuple, att_state_feed)

    # ==================== create init state ==================
    # lstm init state
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    batch_size = tf.shape(init_h)[0]

    init_att = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = (init_c, init_h, init_att)
    concat_op(init_state, axis=1, name="initial_state")  # need to fetch

    # ==================== forward pass ========================
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))
    word_embed = tf.nn.embedding_lookup(word_map, inputs)

    with tf.variable_scope('RNN'):
        outputs, multi_state = multi_cell(tf.squeeze(word_embed, squeeze_dims=[1]),
                                          state=multi_cell_state_feed)

    # ==================== concat states =========================
    lstm_state, att_state = multi_state
    state_c, state_h = lstm_state
    concat_op((state_c, state_h, att_state), axis=1, name="state")  # need to fetch

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    prob = tf.nn.softmax(logits, name="softmax")  # need to fetch
    return prob


# *****************  Beam Search GRAPH *******************************
def _build_tf_beam_inference_decoder(glb_ctx, im, ans_embed, vocab_size,
                                     num_cells, start_token_id, pad_token):
    beam_size = 3
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_lstm_state = concat_op([init_c, init_h], axis=1)
    batch_size = tf.shape(init_h)[0]
    att_zero_state = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = concat_op([init_lstm_state, att_zero_state], axis=1)

    # build LSTM cell and RNN
    lstm_cell = BasicLSTMCell(num_cells, state_is_tuple=False)

    # replicate context of the attention module
    im_shape = im.get_shape().as_list()[1:]
    im = tf.expand_dims(im, 1)  # add a time dim
    im = tf.reshape(tf.tile(im, [1, beam_size, 1, 1, 1]), [-1] + im_shape)
    ans_embed = tf.expand_dims(ans_embed, 1)
    ans_embed_dim = ans_embed.get_shape().as_list()[-1]
    ans_embed = tf.reshape(tf.tile(ans_embed, [1, beam_size, 1]), [-1, ans_embed_dim])

    attention_cell = MultiModalAttentionCell(512, im, ans_embed, keep_prob=1.0)
    multi_cell = MultiRNNCell([lstm_cell, attention_cell], state_is_tuple=False)

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
                                  output_dense=True,
                                  scope='RNN')
    return scores, pathes
