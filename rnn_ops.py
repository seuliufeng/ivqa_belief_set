from rnn_compact_ops import *


def _create_drop_lstm_cell(num_cell, input_keep_prob, output_keep_prob,
                           cell_fn=BasicLSTMCell):
    cell = cell_fn(num_cell)
    return DropoutWrapper(cell, input_keep_prob,
                          output_keep_prob)


def create_drop_lstm_cell(num_cell, input_keep_prob, output_keep_prob,
                          cell_fn=BasicLSTMCell):
    cell = cell_fn(num_cell)
    return DropoutWrapper(cell, input_keep_prob,
                          output_keep_prob)


def build_caption_inputs_and_targets(capt, capt_len):
    max_len = tf.shape(capt)[1]
    # inputs
    inputs = tf.slice(capt, [0, 0], size=[-1, max_len - 1])
    # targets
    targets = tf.slice(capt, [0, 1], size=[-1, max_len - 1])
    # length
    length = capt_len - 1
    return inputs, targets, length
