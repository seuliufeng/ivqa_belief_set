import numpy as np
from config import VOCAB_CONFIG

_START_TOKEN_ID = VOCAB_CONFIG.start_token_id
_END_TOKEN_ID = VOCAB_CONFIG.end_token_id


def padding_or_clip_sequence(seq, seq_len, max_len):
    seq_len = np.minimum(seq_len, max_len)
    batch_size, seq_max_len = seq.shape
    if seq_max_len < max_len:  # padding
        pad_len = max_len - seq_max_len
        pad_seq = np.zeros([batch_size, pad_len], dtype=np.int32)
        output_seq = np.concatenate([seq, pad_seq], axis=1)
    elif seq_max_len > max_len:  # clip
        output_seq = seq[:, :max_len]
    else:
        output_seq = seq
    return output_seq, seq_len


def process_input_data(inputs, pad_token, start_token_id=_START_TOKEN_ID,
                       end_token_id=_END_TOKEN_ID):
    """
    Add start and end tokens to the original inputs
    :param inputs: a numpy array of tokens, without start and end token
    :param pad_token: the token used for padding, padding token is used
    for the computation of loss masks
    :return: numpy array of tokens, with start token, end token, and padding
    tokens added
    """
    im, capt, capt_len = inputs
    capt_len = capt_len.flatten()
    pad = np.ones(shape=[capt_len.size, 1], dtype=np.int32)
    capt = np.concatenate((start_token_id * pad, capt, pad), axis=1)
    capt_len += 2
    for x, x_len in zip(capt, capt_len):
        x[x_len - 1] = end_token_id
        x[x_len:] = pad_token
    return [im, capt, capt_len]


def replace_pad_token(inputs, pad_token):
    im, capt, capt_len = inputs
    capt_len = capt_len.flatten()
    for x, x_len in zip(capt, capt_len):
        x[x_len - 1] = _END_TOKEN_ID
        x[x_len:] = pad_token
    return [im, capt, capt_len]


def add_pad_token(capt, capt_len, pad_token):
    capt_len = capt_len.flatten()
    for x, x_len in zip(capt, capt_len):
        x[x_len:] = pad_token
    return capt, capt_len
