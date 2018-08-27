from config import VOCAB_CONFIG
import numpy as np
import pdb


END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id


def find_unique_rows(scores, pathes):
    sorted_data = pathes[np.lexsort(pathes.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    pathes = sorted_data[row_mask]
    scores = np.zeros_like(pathes, dtype=np.float32)
    return scores, pathes


def put_to_array(sentences):
    sentence_lengths = [len(s) for s in sentences]
    max_length = max(sentence_lengths)
    batch_size = len(sentences)
    token_arrays = np.zeros([batch_size, max_length], dtype=np.int32)
    for s, s_len, target in zip(sentences, sentence_lengths, token_arrays):
        target[:s_len] = s
    token_lens = np.array(sentence_lengths, dtype=np.int32)
    return token_arrays.astype(np.int32), token_lens


def extract_gt(capt, capt_len):
    gt = []
    for c, c_len in zip(capt, capt_len):
        tmp = c[:c_len].tolist()
        gt.append(np.array(tmp))
    return gt


def post_process_prediction(scores, pathes):
    is_end_token = np.equal(pathes, END_TOKEN)
    pred_len = np.argmax(is_end_token, axis=1)
    exceed_max_len = np.logical_not(np.any(is_end_token, axis=1))
    pred_len[exceed_max_len] = pathes.shape[1]

    confs, vpathes = [], []
    for path, score, plen in zip(pathes, scores, pred_len):
        conf = score
        seq = path.tolist()[:plen]
        confs.append(conf)
        vpathes.append(seq)
    return confs, vpathes


def sample_unique_questions(scores, pathes):
    scores, pathes = post_process_prediction(scores, pathes)
    pathes, pathes_len = put_to_array(pathes)
    scores_, pathes_ = find_unique_rows(scores, pathes)
    scores, pathes = post_process_prediction(scores_, pathes_[:, 1:])
    pdb.set_trace()
    return scores, pathes