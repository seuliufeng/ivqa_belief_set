import tensorflow as tf
from rnn_compact_ops import RNNCell, LSTMCell, LSTMStateTuple
import tensorflow.contrib.slim as slim
from ops import concat_fusion, mlb, _soft_attention_pool_with_map, split_op, concat_op
from util import create_dropout_lstm_cells, create_dropout_basic_lstm_cells


def conditional_attention_cell_helper(im, a, part_q, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "ConditionalAttentionCell"
    _, h, w, c = im.get_shape().as_list()
    with tf.variable_scope(scope):
        # QA joint embedding
        ctx = concat_fusion(part_q, a, embed_dim)
        # soft attention
        im_ctx = mlb(im, ctx, embed_dim, keep_prob, scope='Matching')
        v, am = _soft_attention_pool_with_map(im, im_ctx)
        am = tf.reshape(am, shape=[-1, h * w])
    return v, ctx, am


def show_attend_tell_attention_helper(im, part_q, keep_prob, scope=""):
    scope = scope or "ShowAttendTellCell"
    _, h, w, c = im.get_shape().as_list()
    with tf.variable_scope(scope):
        # concat im and part q
        part_q = tf.expand_dims(tf.expand_dims(part_q, 1), 2)
        part_q_tile = tf.tile(part_q, [1, h, w, 1])
        im_pq = concat_op([im, part_q_tile], axis=3)
        im_ctx = slim.conv2d(im_pq, 512, 1, activation_fn=tf.nn.tanh,
                             scope='vq_fusion')
        im_ctx = slim.dropout(im_ctx, keep_prob=keep_prob)
        v, _ = _soft_attention_pool_with_map(im, im_ctx)
    return v


class MultiModalAttentionCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, embed_dim, visual_ctx, answer_ctx, input_size=None,
                 keep_prob=1.0,
                 activation=tf.tanh):
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        _, h, w, num_units = visual_ctx.get_shape().as_list()
        self._num_grids = h * w
        self._num_units = embed_dim
        self._activation = activation
        self._keep_prob = keep_prob
        self._embed_dim = embed_dim
        self._context = visual_ctx  # visual context, e.g., res5c
        self._answer_context = answer_ctx  # answer embedding, used to get joint embedding with partial questions

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Attention cell with answer context."""
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('Attention'):
                v, ctx, am = conditional_attention_cell_helper(self._context,
                                                               self._answer_context,
                                                               inputs,
                                                               self._embed_dim,
                                                               keep_prob=self._keep_prob)
                h = mlb(v, ctx, self._embed_dim, self._keep_prob, scope='OutputMLB')
                # residual connection
                h = inputs + h
        return h, h


class RerankAttentionCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, embed_dim, visual_ctx, answer_ctx, input_size=None,
                 keep_prob=1.0,
                 activation=tf.tanh):
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        _, h, w, num_units = visual_ctx.get_shape().as_list()
        self._num_grids = h * w
        self._num_units = embed_dim
        self._activation = activation
        self._keep_prob = keep_prob
        self._embed_dim = embed_dim
        self._context = visual_ctx  # visual context, e.g., res5c
        self._answer_context = answer_ctx  # answer embedding, used to get joint embedding with partial questions

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='MultiModalAttentionCell'):
        """Attention cell with answer context."""
        with tf.variable_scope('MultiModalAttentionCell'):
            with tf.variable_scope('Attention'):
                v, ctx, am = conditional_attention_cell_helper(self._context,
                                                               self._answer_context,
                                                               inputs,
                                                               self._embed_dim,
                                                               keep_prob=self._keep_prob)
                h = mlb(v, ctx, self._embed_dim, self._keep_prob, scope='OutputMLB')
                # residual connection
                new_h = inputs + h
        # new_h, new_state
        return new_h, h


class ShowAttendTellCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, embed_dim, visual_ctx,
                 keep_prob=1.0,
                 activation=tf.tanh,
                 state_is_tuple=True):
        _, h, w, num_units = visual_ctx.get_shape().as_list()
        self._num_grids = h * w
        self._num_units = embed_dim
        self._activation = activation
        self._keep_prob = keep_prob
        self._embed_dim = embed_dim
        self._context = visual_ctx  # visual context, e.g., res5c
        self._state_is_tuple = state_is_tuple
        self._lstm_cell = create_dropout_basic_lstm_cells(self._num_units,
                                                          input_keep_prob=self._keep_prob,
                                                          output_keep_prob=self._keep_prob,
                                                          state_is_tuple=self._state_is_tuple)

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Attention cell with answer context."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                _, h = state
            else:
                _, h = split_op(values=state, num_splits=2, axis=1)

            with tf.variable_scope('Attention'):
                v = show_attend_tell_attention_helper(self._context, h, self._keep_prob)
            lstm_input = concat_op([v, inputs], axis=1)
            lstm_h, next_state = self._lstm_cell(lstm_input, state=state)
            # concat outputs
            h_ctx_reduct = concat_fusion(v, lstm_h, self._num_units, act_fn=None)
            output = h_ctx_reduct + inputs
        return output, next_state
