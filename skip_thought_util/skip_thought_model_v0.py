import tensorflow as tf
import os
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from util import get_default_initializer, create_init_variables_op


SKIP_THOUGHT_DIM = 2400
SKIP_THOUGHT_WORD_DIM = 620


class SkipThoughtGRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell implementation in Skip Thought Vectors."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, _linear([inputs, state],
                                                     2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("CandidateW"):
                wx = _linear(inputs, self._num_units, True)
            with vs.variable_scope("CandidateU"):
                wu = _linear(state, self._num_units, False)
            c = self._activation(wx + r * wu)
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class SkipThoughtEncoder(object):
    def __init__(self, vocab_size, keep_prob=1.0,
                 use_word_embed=False, use_outputs=False,
                 type='default'):
        self._vocab_size = vocab_size
        self._word_embed_dim = 620
        self._num_gru_cells = 2400
        self._l2_normalise = False
        self._graph_created = False
        self._use_outputs = use_outputs
        self._use_default = type == 'default'
        self._word_embedding = None
        self._keep_prob = keep_prob
        self.init_fn = None
        self._cells = None
        self._return_word_embed = use_word_embed
        if self._use_default:
            self._gru_fn = SkipThoughtGRUCell
        else:
            self._gru_fn = tf.nn.rnn_cell.GRUCell
        self.build_model()

    def build_model(self):
        # with tf.variable_scope("SkipThought/word_embedding"), tf.device("/cpu:0"):
        with tf.variable_scope("SkipThought/word_embedding"):
            self._word_embedding = tf.get_variable(
                name="map",
                shape=[self._vocab_size, self._word_embed_dim],
                initializer=get_default_initializer())
        cells = self._gru_fn(self._num_gru_cells)
        self._cells = tf.nn.rnn_cell.DropoutWrapper(cells, input_keep_prob=self._keep_prob,
                                                    output_keep_prob=self._keep_prob)

    def __call__(self, inputs, input_len):
        word_embedding = tf.nn.embedding_lookup(self._word_embedding, inputs)
        outputs, state = tf.nn.dynamic_rnn(self._cells, word_embedding, input_len,
                                     dtype=tf.float32, scope='SkipThought')
        self._graph_created = True
        if self._use_outputs and self._return_word_embed:
            return word_embedding, outputs, state
        if self._return_word_embed:
            return word_embedding, state
        if self._use_outputs:
            return outputs, state
        else:
            return state

    def setup_initializer(self, ckpt_path):
        if not self._graph_created:
            raise Exception('Init OP should be called after graph is created')
        var_list = [var for var in tf.trainable_variables() if 'SkipThought' in var.name]
        init_ops = create_init_variables_op(ckpt_path, var_list)

        def restore_fn(sess):
            tf.logging.info('Restoring Skip Thought variables from file %s'
                            % os.path.basename(ckpt_path))
            sess.run(init_ops)

        self.init_fn = restore_fn
        return restore_fn


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
