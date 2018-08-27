import tensorflow as tf
import numpy as np
from util import load_hdf5, save_hdf5

tf.flags.DEFINE_string("testset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_boolean("use_var", True,
                        "Use variational VQA or VQA.")
FLAGS = tf.flags.FLAGS


def process(K=1., use_global_thresh=False):
    _model_suffix = 'var_' if FLAGS.use_var else ''
    # load VQA scores
    d = load_hdf5('data4/%svqg_%s_qa_scores.data' % (_model_suffix, FLAGS.testset))
    # cand_scores = d['ext_cand_scores']
    quest_ids = d['ext_quest_ids']
    ext_top_answer = d['ext_cand_pred_labels']
    cand_scores = d['ext_cand_pred_scores']

    # load QAs
    d = load_hdf5('data4/%svqg_%s_question_tokens.data' % (_model_suffix, FLAGS.testset))
    ext_quest_arr = d['ext_quest_arr']
    ext_quest_len = d['ext_quest_len']
    seed_quest_ids = d['ext_quest_ids']
    # ext_top_answer = d['ext_top_answer']

    assert (np.all(np.equal(quest_ids, seed_quest_ids)))

    num_all = quest_ids.shape[0]
    print(quest_ids.shape[0])

    # build index
    quest_id2index = {}
    for i, quest_id_tuple in enumerate(quest_ids):
        quest_id, _ = quest_id_tuple.tolist()
        if quest_id in quest_id2index:
            quest_id2index[quest_id].append(i)
        else:
            quest_id2index[quest_id] = [i]

    # parse
    slice_index = []
    unk_quest_ids = quest_id2index.keys()
    num = len(unk_quest_ids)

    if use_global_thresh:
        loc = int(num * K)
        thresh = -np.sort(-cand_scores)[loc]
        thresh = 0.3
        print('Global thresh: %0.2f' % thresh)
        keep_tab = cand_scores > thresh
        quest_ids = quest_ids[keep_tab]
        ext_quest_arr = ext_quest_arr[keep_tab]
        ext_quest_len = ext_quest_len[keep_tab]
        ext_top_answer = ext_top_answer[keep_tab]
    else:
        for i, quest_id in enumerate(unk_quest_ids):
            if i % 1000 == 0:
                print('Processed %d/%d' % (i, num))
            _index = quest_id2index[quest_id]
            _scores = cand_scores[_index]
            _max_score_idx = (-_scores).argsort()[:K]
            if K == 1:
                add_idx = _index[_max_score_idx]
                slice_index.append(add_idx)
            else:
                add_idx = [_index[_mci] for _mci in _max_score_idx]
                slice_index += add_idx

        # get data
        quest_ids = quest_ids[slice_index]
        ext_quest_arr = ext_quest_arr[slice_index]
        ext_quest_len = ext_quest_len[slice_index]
        ext_top_answer = ext_top_answer[slice_index]

    # save
    save_hdf5('data4/%svqg_%s_question_answers_fltmax.data' % (_model_suffix, FLAGS.testset),
              {'ext_quest_arr': ext_quest_arr,
               'ext_quest_len': ext_quest_len,
               'ext_quest_ids': quest_ids,
               'ext_top_answer': ext_top_answer})
    print('%d/%d' % (ext_top_answer.size, num_all))


if __name__ == '__main__':
    process(K=15., use_global_thresh=True)
