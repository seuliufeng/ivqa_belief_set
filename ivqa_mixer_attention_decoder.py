import tensorflow as tf
import tensorflow.contrib.slim as slim
from rnn_ops import create_drop_lstm_cell
from rnn_ops import build_caption_inputs_and_targets
from random_decoding import create_greedy_decoder
from ops import concat_op, split_op
from rnn_compact_ops import *
from ivqa_rnn_cells import MultiModalAttentionCell
from beam_search_util.tf_beam_decoder import beam_decoder
from config import VOCAB_CONFIG


def build_decoder(im, attr, ans_embed, quest, quest_len, rewards, advantage,
                  vocab_size, keep_prob, pad_token, num_dec_cells, phase='train',
                  reuse=False, xe_mask=None, T=4):
    if attr is None:
        in_embed = ans_embed
    else:
        in_embed = concat_op(values=[attr, ans_embed], axis=1)

    with tf.variable_scope('inverse_vqa', reuse=reuse):
        if phase == 'train' or phase == 'condition' or phase == 'evaluate':
            inputs, targets, length = build_caption_inputs_and_targets(quest,
                                                                       quest_len)
            return _build_training_decoder(in_embed, im, ans_embed, inputs, length, targets,
                                           vocab_size, num_dec_cells, keep_prob,
                                           pad_token, rewards, advantage, xe_mask)
        elif phase == 'random':
            return _build_random_inference_decoder(in_embed, im, ans_embed,
                                                   vocab_size, num_dec_cells,
                                                   VOCAB_CONFIG.start_token_id,
                                                   pad_token)
        elif phase == 'mixer':
            return _build_mixer_inference_decoder(in_embed, im, ans_embed, vocab_size,
                                                  num_dec_cells, quest, quest_len,
                                                  pad_token, T)
        elif phase == 'beam':
            return _build_tf_beam_inference_decoder(in_embed, im, ans_embed,
                                                    vocab_size, num_dec_cells,
                                                    VOCAB_CONFIG.start_token_id,
                                                    pad_token)
        else:
            raise Exception('unknown option')


# *****************  TRAINING GRAPH *******************************
def _build_training_decoder(glb_ctx, im, ans, inputs, length, targets, vocab_size,
                            num_cells, keep_prob, pad_token, rewards, advantage, xe_mask):
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

    attention_cell = MultiModalAttentionCell(512, im, ans, keep_prob=keep_prob)
    attention_cell = DropoutWrapper(attention_cell, input_keep_prob=1.0,
                                    output_keep_prob=keep_prob)

    multi_cell = MultiRNNCell([lstm, attention_cell], state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(multi_cell, inputs, length, initial_state=init_state,
                                   dtype=tf.float32, scope='RNN')

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')

    # compute loss
    batch_size = tf.shape(targets)[0]
    targets = tf.reshape(targets, [-1])
    valid_mask = tf.not_equal(targets, pad_token)
    valid_xe_mask = tf.cast(tf.logical_and(xe_mask, valid_mask), tf.float32)
    valid_rl_mask = tf.cast(tf.logical_and(tf.logical_not(xe_mask), valid_mask),
                            tf.float32)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=targets)

    # compute cross entropy loss
    xe_loss = tf.div(tf.reduce_sum(losses * valid_xe_mask),
                     tf.reduce_sum(valid_xe_mask), name='xe_loss')
    slim.losses.add_loss(xe_loss)

    # compute reinforce loss
    advantage = tf.reshape(advantage, [-1])
    actor_loss = tf.div(tf.reduce_sum(losses * valid_rl_mask * advantage),
                        tf.reduce_sum(valid_rl_mask),
                        name='actor_loss')
    slim.losses.add_loss(actor_loss)
    return tf.reshape(losses * tf.cast(valid_mask, tf.float32), [batch_size, -1])


# *****************  CRITIC GRAPH *******************************
def build_critic(glb_ctx, im, ans, quest, quest_len, vocab_size,
                 num_cells, pad_token, rewards, xe_mask):
    # process inputs
    inputs, targets, length = build_caption_inputs_and_targets(quest,
                                                               quest_len)

    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)

    # init state / image embedding
    ctx = concat_op([glb_ctx, im, ans], axis=1)
    with tf.variable_scope('critic'):
        init_h = slim.fully_connected(ctx, num_cells, activation_fn=tf.nn.tanh,
                                      scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_state = LSTMStateTuple(init_c, init_h)

    # word embedding
    with tf.variable_scope('inverse_vqa', reuse=True):
        with tf.variable_scope('word_embedding'):
            word_map = tf.get_variable(
                name="word_map",
                shape=[vocab_size, num_cells],
                initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                          dtype=tf.float32))
    inputs = tf.nn.embedding_lookup(word_map, inputs)
    inputs = tf.stop_gradient(inputs)

    # build LSTM cell and RNN
    lstm = BasicLSTMCell(num_cells)

    with tf.variable_scope('critic'):
        outputs, _ = tf.nn.dynamic_rnn(lstm, inputs, length, initial_state=init_state,
                                       dtype=tf.float32, scope='rnn')

        # compute critic
        values = slim.fully_connected(outputs, 1, activation_fn=None, scope='value')
        values = tf.reshape(values, [-1])

    # compute loss
    targets = tf.reshape(targets, [-1])
    valid_mask = tf.not_equal(targets, pad_token)
    rl_mask = tf.logical_not(tf.reshape(xe_mask, [-1]))
    critic_mask = tf.logical_and(rl_mask, valid_mask)
    critic_mask = tf.cast(critic_mask, tf.float32)
    rewards = tf.reshape(rewards, [-1])

    # compute loss
    critic_loss = tf.div(tf.reduce_sum(tf.square(values - rewards) * critic_mask * 0.5),
                         tf.reduce_sum(critic_mask), name='critic_loss')
    slim.losses.add_loss(critic_loss)
    return values


# *****************  Greedy Decoding GRAPH ***************************
def _build_random_inference_decoder(glb_ctx, im, ans_embed, vocab_size,
                                    num_cells, start_token_id, pad_token):
    keep_prob = 0.7
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_lstm_state = LSTMStateTuple(init_c, init_h)
    batch_size = tf.shape(init_h)[0]
    att_zero_state = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = (init_lstm_state, att_zero_state)

    # build LSTM cell and RNN
    lstm_cell = create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                      output_keep_prob=keep_prob,
                                      cell_fn=BasicLSTMCell)
    attention_cell = MultiModalAttentionCell(512, im, ans_embed, keep_prob=1.0)
    attention_cell = DropoutWrapper(attention_cell, input_keep_prob=1.0,
                                    output_keep_prob=keep_prob)
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
                                 softmax_params,
                                 start_token_id)


# *****************  Greedy Decoding GRAPH ***************************
def _build_mixer_inference_decoder(glb_ctx, im, ans_embed, vocab_size,
                                   num_cells, capt, capt_len,
                                   pad_token, n_free_run_steps):
    keep_prob = 0.7
    # avoid out of range error
    vocab_size = max(vocab_size, pad_token + 1)
    # init state / image embedding
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    init_lstm_state = LSTMStateTuple(init_c, init_h)

    # zero attention cell state
    batch_size = tf.shape(init_h)[0]
    att_zero_state = tf.zeros([batch_size, num_cells], dtype=tf.float32)

    # build LSTM cell and RNN
    lstm_cell = create_drop_lstm_cell(num_cells, input_keep_prob=keep_prob,
                                      output_keep_prob=keep_prob,
                                      cell_fn=BasicLSTMCell)
    attention_cell = MultiModalAttentionCell(512, im, ans_embed, keep_prob=1.0)
    attention_cell = DropoutWrapper(attention_cell, input_keep_prob=1.0,
                                    output_keep_prob=keep_prob)
    multi_cell = MultiRNNCell([lstm_cell, attention_cell], state_is_tuple=True)

    # word embedding
    with tf.variable_scope('word_embedding'):
        word_map = tf.get_variable(
            name="word_map",
            shape=[vocab_size, num_cells],
            initializer=tf.random_uniform_initializer(-0.08, 0.08,
                                                      dtype=tf.float32))

    # run teacher forcing steps
    inputs = tf.nn.embedding_lookup(word_map, capt)
    if type(n_free_run_steps) != tf.Tensor:
        n_free_run_steps = tf.constant(n_free_run_steps, dtype=tf.int32)

    n_teacher_forcing_steps = tf.maximum(capt_len - n_free_run_steps,
                                         tf.constant(0, dtype=tf.int32))

    def gather_by_col(arr, cols):
        batch_size = tf.shape(arr)[0]
        num_cols = tf.shape(arr)[1]
        index = tf.range(batch_size) * num_cols + cols
        arr = tf.reshape(arr, [-1])
        return tf.reshape(tf.gather(arr, index), [batch_size])

    free_run_start_tokens = gather_by_col(capt, n_teacher_forcing_steps)

    with tf.variable_scope('RNN'):
        with tf.variable_scope('multi_rnn_cell'):
            with tf.variable_scope('cell_0') as sc:
                _, lstm_state = tf.nn.dynamic_rnn(lstm_cell, inputs,
                                                  n_teacher_forcing_steps,
                                                  initial_state=init_lstm_state,
                                                  dtype=tf.float32, scope=sc)
                dummy_hideen = lstm_state[0]
            with tf.variable_scope('cell_1'):
                attention_cell(dummy_hideen, att_zero_state)
    init_state = (lstm_state, att_zero_state)

    # fetch start tokens

    # apply weights for outputs
    with tf.variable_scope('logits'):
        weights = tf.get_variable('weights', shape=[num_cells, vocab_size], dtype=tf.float32)
        biases = tf.get_variable('biases', shape=[vocab_size])
        softmax_params = [weights, biases]

    path = create_greedy_decoder(init_state, multi_cell, word_map,
                                 softmax_params,
                                 free_run_start_tokens)
    return path, free_run_start_tokens
    # return create_greedy_decoder(init_state, multi_cell, word_map,
    #                              softmax_params,
    #                              free_run_start_tokens)


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
