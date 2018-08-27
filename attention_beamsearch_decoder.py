import tensorflow as tf
import tensorflow.contrib.slim as slim
from attention_cells import ConditionalSoftAttentionCell, ConditionalSoftAttentionCell2, \
    ConditionalSoftAttentionCell3, ConditionalSoftAttentionCell4, ConditionalSoftAttentionCell5, \
    BaselineSoftAttentionCell


def build_vaq_dec_attention_predictor(glb_ctx, im, ans, inputs, vocab_size,
                                      num_cells, pad_token, cell_option):
    vocab_size = max(vocab_size, pad_token + 1)

    # =================== create cells ========================
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_cells)
    if cell_option == 2:
        attention_cell = ConditionalSoftAttentionCell2(512, im, ans, keep_prob=1.0)
    elif cell_option == 3:
        attention_cell = ConditionalSoftAttentionCell3(512, im, ans, keep_prob=1.0)
    elif cell_option == 4:
        attention_cell = ConditionalSoftAttentionCell4(512, im, ans, keep_prob=1.0)
    elif cell_option == 5:
        attention_cell = ConditionalSoftAttentionCell5(512, im, ans, keep_prob=1.0)
    elif cell_option == 6:  # baseline
        attention_cell = BaselineSoftAttentionCell(512, im, ans, keep_prob=1.0)
    else:
        attention_cell = ConditionalSoftAttentionCell(512, im, ans, keep_prob=1.0)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm, attention_cell],
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
    feed_c, feed_h = tf.split(1, 2, lstm_state_feed)
    state_tuple = tf.nn.rnn_cell.LSTMStateTuple(feed_c, feed_h)
    multi_cell_state_feed = (state_tuple, att_state_feed)

    # ==================== create init state ==================
    # lstm init state
    init_h = slim.fully_connected(glb_ctx, num_cells, activation_fn=tf.nn.tanh,
                                  scope='init_h')
    init_c = tf.zeros_like(init_h)
    batch_size = tf.shape(init_h)[0]

    init_att = tf.zeros([batch_size, num_cells], dtype=tf.float32)
    init_state = (init_c, init_h, init_att)
    tf.concat(1, init_state, name="initial_state")  # need to fetch

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
    tf.concat(1, (state_c, state_h, att_state), name="state")  # need to fetch

    # predict next word
    outputs = tf.reshape(outputs, [-1, num_cells])
    logits = slim.fully_connected(outputs, vocab_size, activation_fn=None,
                                  scope='logits')
    prob = tf.nn.softmax(logits, name="softmax")  # need to fetch
    return prob
