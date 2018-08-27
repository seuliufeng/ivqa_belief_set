import numpy as np
from config import VOCAB_CONFIG

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id


def put_to_array(sentences, pad_token=None, max_length=None):
    pad_token = pad_token or 0
    sentence_lengths = np.array([len(s) for s in sentences])
    if max_length is None:
        max_length = sentence_lengths.max()
    else:
        sentence_lengths[sentence_lengths > max_length] = max_length
    batch_size = len(sentences)
    token_arrays = np.ones([batch_size, max_length], dtype=np.int32) * pad_token
    for s, s_len, target in zip(sentences, sentence_lengths, token_arrays):
        target[:s_len] = s[:s_len]
    # token_lens = np.array(sentence_lengths, dtype=np.int32)
    return [token_arrays.astype(np.int32), sentence_lengths.astype(np.int32)]


def post_process_prediction(scores, pathes, add_start_end=True, do_sum=True):
    assert (add_start_end)
    is_end_token = np.equal(pathes, END_TOKEN)
    pred_len = np.argmax(is_end_token, axis=1)
    exceed_max_len = np.logical_not(np.any(is_end_token, axis=1))
    pred_len[exceed_max_len] = pathes.shape[1]

    confs, vpathes = [], []
    for path, score, plen, exceed in zip(pathes, scores, pred_len,
                                         exceed_max_len):
        seq = path.tolist()[:plen]
        if add_start_end:
            seq = [START_TOKEN] + seq
        if exceed:
            conf = score[:plen].sum() if do_sum else score[0]
        else:
            conf = score[:plen + 1].sum() if do_sum else score[0]
            seq += [END_TOKEN]
        confs.append(conf)
        vpathes.append(seq)
    return confs, vpathes


def find_unique_pathes(scores, pathes, noise=None, use_count=False):
    path_dict = {}
    for idx, p in enumerate(pathes):
        path_key = ' '.join([str(pi) for pi in p])
        if path_key in path_dict:
            path_dict[path_key].append(idx)
        else:
            path_dict[path_key] = [idx]
    # get unique path and scores
    new_pathes, new_scores, new_noise = [], [], []
    n_votes = []
    for v in path_dict.values():
        new_pathes.append(pathes[v[0]])
        cur_scores = [scores[_idx] for _idx in v]
        new_scores.append(max(cur_scores))
        n_votes.append(len(v))
        if noise is not None:
            _pick_idx = np.argmax(cur_scores)
            assert (cur_scores[_pick_idx] == new_scores[-1])
            new_noise.append(noise[_pick_idx])
    outputs = [new_scores, new_pathes]
    if new_noise:
        outputs.append(new_noise)
    if use_count:
        outputs.append(n_votes)
    return outputs


def post_process_variation_questions(scores, pathes, _this_batch_size):
    ivqa_scores, ivqa_pathes = [], []
    # parse sentences
    scores, pathes = post_process_prediction(scores, pathes,
                                             add_start_end=True)
    # process for each sample
    num_sampled = int(len(pathes) / _this_batch_size)
    _noise_offset = np.arange(0, num_sampled, dtype=np.int32) * _this_batch_size
    for _s_id in range(_this_batch_size):
        _index = _noise_offset + _s_id
        cur_scores = [scores[_idx] for _idx in _index]
        cur_pathes = [pathes[_idx] for _idx in _index]

        cur_scores, cur_pathes = find_unique_pathes(cur_scores, cur_pathes)
        ivqa_scores.append(cur_scores)
        ivqa_pathes.append(cur_pathes)
    return ivqa_scores, ivqa_pathes


def post_process_variation_questions_with_count(scores, pathes, _this_batch_size):
    ivqa_scores, ivqa_pathes, ivqa_counts = [], [], []
    # parse sentences
    scores, pathes = post_process_prediction(scores, pathes,
                                             add_start_end=True)
    # process for each sample
    num_sampled = int(len(pathes) / _this_batch_size)
    _noise_offset = np.arange(0, num_sampled, dtype=np.int32) * _this_batch_size
    for _s_id in range(_this_batch_size):
        _index = _noise_offset + _s_id
        cur_scores = [scores[_idx] for _idx in _index]
        cur_pathes = [pathes[_idx] for _idx in _index]

        cur_scores, cur_pathes, cur_count = find_unique_pathes(cur_scores, cur_pathes, use_count=True)
        ivqa_scores.append(cur_scores)
        ivqa_pathes.append(cur_pathes)
        ivqa_counts.append(cur_count)
    return ivqa_scores, ivqa_pathes, ivqa_counts


def post_process_variation_questions_with_count_v2(scores, pathes, _this_batch_size):
    ivqa_scores, ivqa_pathes, ivqa_counts = [], [], []
    # parse sentences
    scores, pathes = post_process_prediction(scores, pathes,
                                             add_start_end=True,
                                             do_sum=False)
    # process for each sample
    num_sampled = int(len(pathes) / _this_batch_size)
    _noise_offset = np.arange(0, num_sampled, dtype=np.int32) * _this_batch_size
    for _s_id in range(_this_batch_size):
        _index = _noise_offset + _s_id
        cur_scores = [scores[_idx] for _idx in _index]
        cur_pathes = [pathes[_idx] for _idx in _index]

        cur_scores, cur_pathes, cur_count = find_unique_pathes(cur_scores, cur_pathes, use_count=True)
        ivqa_scores.append(cur_scores)
        ivqa_pathes.append(cur_pathes)
        ivqa_counts.append(cur_count)
    return ivqa_scores, ivqa_pathes, ivqa_counts


def process_one(scores, pathes):
    def shape_equal(shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        for d1, d2 in zip(shape1, shape2):
            if d1 != d2:
                return False
        return True

    # check shape
    if not shape_equal(scores.shape, pathes.shape):
        scores = np.tile(scores[:, np.newaxis], [1, pathes.shape[1]])
    # extract pathes
    scores, pathes = post_process_prediction(scores, pathes)
    # find unique
    return find_unique_pathes(scores, pathes)


def post_process_variation_questions_noise(scores, pathes, noise, _this_batch_size,
                                           find_unique=True):
    ivqa_scores, ivqa_pathes, ivqa_noise = [], [], []
    # parse sentences
    scores, pathes = post_process_prediction(scores, pathes,
                                             add_start_end=True)
    # process for each sample
    num_sampled = int(len(pathes) / _this_batch_size)
    _noise_offset = np.arange(0, num_sampled, dtype=np.int32) * _this_batch_size
    for _s_id in range(_this_batch_size):
        _index = _noise_offset + _s_id
        cur_scores = [scores[_idx] for _idx in _index]
        cur_pathes = [pathes[_idx] for _idx in _index]
        cur_noise = noise[_index]

        if find_unique:
            cur_scores, cur_pathes, cur_noise = find_unique_pathes(cur_scores, cur_pathes, cur_noise)
        ivqa_scores.append(cur_scores)
        ivqa_pathes.append(cur_pathes)
        ivqa_noise.append(cur_noise)
    return ivqa_scores, ivqa_pathes, ivqa_noise


def _parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen].tolist())
    return seqs


def wrap_samples_for_language_model(sampled=None, pad_token=0, gts=None,
                                    max_length=None):
    pathes = []
    for ps in sampled:
        for p in ps:
            pathes.append(p[1:])  # keep end token
    outputs = put_to_array(pathes, pad_token, max_length=max_length)
    if gts is not None:
        gt_pathes = _parse_gt_questions(*gts)
        for _gt in gt_pathes:
            _gt.append(2)
        _gt = put_to_array(gt_pathes, pad_token, max_length=max_length)
        outputs += _gt
    return outputs


def wrap_samples_for_language_model_v2(sampled=None, pad_token=0,
                                       max_length=None):
    pathes = []
    for ps in sampled:
        for p in ps:
            pathes.append(p[1:])  # keep end token
    outputs = put_to_array(pathes, pad_token, max_length=max_length)
    return outputs, pathes


def prepare_reinforce_data_max(pathes, noise, rewards, pad_token=None):
    idx = 0
    max_noise, max_paths, max_rewards = [], [], []
    for _var_s, _var_n in zip(pathes, noise):
        _n = len(_var_s)
        ind = np.arange(idx, idx + _n, 1)
        _max_reward_idx = rewards[ind].argmax()
        _max_p = _var_s[_max_reward_idx]
        _max_n = _var_n[_max_reward_idx]
        _max_r = rewards[ind][_max_reward_idx]
        max_noise.append(_max_n[np.newaxis, :])
        max_paths.append(_max_p)
        max_rewards.append(_max_r)
        idx += _n
    max_noise = np.concatenate(max_noise, axis=0).astype(np.float32)
    max_rewards = np.array(max_rewards, dtype=np.float32)
    max_path_arr, max_path_len = put_to_array(max_paths, pad_token)
    max_len = max_path_arr.shape[1]
    max_rewards = np.tile(max_rewards[:, np.newaxis], [1, max_len - 1])
    return max_path_arr, max_path_len, max_noise, max_rewards


def prepare_reinforce_data(pathes, noise, rewards, pad_token=None):
    _pathes, _noise = [], []

    for _var_s, _var_n in zip(pathes, noise):
        _pathes += _var_s
        _noise += [_var_n]
    _noise = np.concatenate(_noise, axis=0).astype(np.float32)
    path_arr, path_len = put_to_array(_pathes, pad_token)
    max_len = path_arr.shape[1]
    _rewards = np.tile(rewards[:, np.newaxis], [1, max_len - 1])

    return path_arr, path_len, _noise, _rewards


def remove_out_of_vocabulary_samples(inputs, is_oov):
    return [_in[is_oov] for _in in inputs]


def correct_language_model_inputs(inputs, is_gt):
    fake, fake_len, real, real_len = inputs
    real = np.concatenate([real, fake[is_gt, :]], axis=0)
    real_len = np.concatenate([real_len, fake_len[is_gt]], axis=0)
    fake_mask = np.logical_not(is_gt)
    fake = fake[fake_mask, :]
    fake_len = fake_len[fake_mask]
    return [fake, fake_len, real, real_len]


def correct_vqa_labels(vqa_labels, rewards_all, is_in_vocab, cider_thresh=0.3):
    vqa, cider, language, diversity, _ = [np.squeeze(v) for v in np.split(rewards_all, 5, axis=1)]
    # positive_mask = cider > 0.3
    negative_mask = cider < 0.002
    hard_target_mask = np.require(cider > 0.3, np.float32)
    legal_mask = np.logical_and(language > 0.1,
                                diversity > 0)
    legal_mask = legal_mask * is_in_vocab
    vqa_labels[negative_mask] = 2000  # mark generated as K+1 class
    return legal_mask, vqa_labels, hard_target_mask


def concat_arrays(arr1, arr2):
    n_arr1, max_d_arr1 = arr1.shape
    n_arr2, max_d_arr2 = arr2.shape
    if max_d_arr1 != max_d_arr2:
        max_d = max(max_d_arr1, max_d_arr2)
        pad_d1 = max_d - max_d_arr1
        pad_d2 = max_d - max_d_arr2
        # padding
        pad_1 = np.zeros([n_arr1, pad_d1], dtype=arr1.dtype)
        arr1 = np.concatenate([arr1, pad_1], 1)
        pad_2 = np.zeros([n_arr2, pad_d2], dtype=arr2.dtype)
        arr2 = np.concatenate([arr2, pad_2], 1)
    # concatenate
    return np.concatenate([arr1, arr2], 0)


def concat_vqa_batch(gt_inputs, aug_inputs):
    outputs = []
    for arr1, arr2 in zip(gt_inputs, aug_inputs):
        try:
            if arr1.ndim == 1:
                arr = np.concatenate([arr1, arr2], axis=0)
            elif arr1.ndim == 2:
                arr = concat_arrays(arr1, arr2)
            else:
                raise Exception('Unsupported dim')
            outputs.append(arr)
        except:
            print('Error')
            import pdb
            pdb.set_trace()
    return outputs


def test_post_process_prediction():
    import pdb
    seq = np.random.randint(low=0, high=100, size=(5, 8))
    # case 1, do not have end token
    row3 = seq[3]
    row3[row3 == END_TOKEN] = END_TOKEN + 1
    # case 2, starts with end token
    row1 = seq[1]
    row1[0] = END_TOKEN
    # case 3, normal case, end token at last column
    row0 = seq[0]
    row0[row0 == END_TOKEN] = END_TOKEN + 1
    row0[-1] = END_TOKEN
    # case 4, normal case, end token in the middle
    row4 = seq[4]
    row4[row4 == END_TOKEN] = END_TOKEN + 1
    row4[4] = END_TOKEN
    print('Pred:')
    print(seq)
    # test 1
    _, pathes = post_process_prediction(seq, seq, add_start_end=True)
    arr, arr_len = put_to_array(pathes)
    print('Processed:')
    print(arr)
    print(arr_len)
    # # test 2
    arr, arr_len = put_to_array(pathes, pad_token=-1)
    print('Processed:')
    print(arr)
    print(arr_len)


if __name__ == '__main__':
    test_post_process_prediction()
