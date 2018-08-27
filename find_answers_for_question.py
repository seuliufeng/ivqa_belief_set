import tensorflow as tf
import numpy as np
from util import load_hdf5, load_json, save_hdf5

tf.flags.DEFINE_string("subset", "kptrain",
                       "Dataset for question extraction.")
tf.flags.DEFINE_boolean("use_var", True,
                        "Use variational VQA or VQA.")
FLAGS = tf.flags.FLAGS


def load_answer_data(subset):
    # get question ids
    d = load_json('data/vqa_std_mscoco_%s.meta' % subset)
    quest_ids = d['quest_id']

    # load top answers
    d = load_hdf5('data/vqa_std_mscoco_%s.data' % subset)
    top_answers = d['answer']

    # load answer sequence
    d = load_hdf5('data/answer_std_mscoco_%s.data' % subset)
    answer_arr = d['ans_arr']
    answer_arr_len = d['ans_len']

    quest_id2index = {qid: i for i, qid in enumerate(quest_ids)}
    return quest_id2index, quest_ids, top_answers, answer_arr, answer_arr_len


def load_question_candidates(subset):
    _model_suffix = 'var_' if FLAGS.use_var else ''
    d = load_hdf5('data4/%sivqa_%s_question_tokens.data' % (_model_suffix,
                                                            subset))
    ext_quest_arr = d['ext_quest_arr']
    ext_quest_len = d['ext_quest_len']
    seed_quest_ids = d['ext_quest_ids']
    quest_id2index, quest_ids, top_answers, \
    answer_arr, answer_arr_len = load_answer_data(subset)

    ext_top_answer = []
    ext_answer_arr = []
    ext_answer_arr_len = []

    # process answer data
    num = seed_quest_ids.shape[0]
    for i, qids in enumerate(seed_quest_ids):
        if i % 1000 == 0:
            print('Processed %d/%d' % (i, num))
        idx = quest_id2index[qids[0]]
        ext_top_answer.append(top_answers[idx])
        ext_answer_arr.append(answer_arr[idx])
        ext_answer_arr_len.append(answer_arr_len[idx])
    # concat data
    ext_top_answer = np.array(ext_top_answer).astype(np.int32)
    ext_answer_arr = np.concatenate(ext_answer_arr).astype(np.int32)
    ext_answer_arr_len = np.array(ext_answer_arr_len).astype(np.int32)
    save_hdf5('data4/%sivqa_%s_question_answers.data' % (_model_suffix,
                                                         FLAGS.subset),
              {'ext_quest_arr': ext_quest_arr,
               'ext_quest_len': ext_quest_len,
               'ext_quest_ids': seed_quest_ids,
               'ext_top_answer': ext_top_answer,
               'ext_answer_arr': ext_answer_arr,
               'ext_answer_arr_len': ext_answer_arr_len})


def main():
    subset = FLAGS.subset
    load_question_candidates(subset)


if __name__ == '__main__':
    main()
