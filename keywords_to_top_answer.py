from util import load_hdf5
import pdb
import numpy as np


def _load_mscoco_dict():
    import cPickle
    with open('/import/vision-ephemeral/fl302/code/mscap/code/vocabs/vocab_train.pkl', 'rb') as f:
        d = cPickle.load(f)
    return d['words']


def load_top_answers():
    top_ans_file = '/import/vision-ephemeral/fl302/code/' \
                   'VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    d = {}
    with open(top_ans_file, 'r') as fs:
        for i, line in enumerate(fs):
            key = line.strip()
            d[key] = i
    return d


def stat_matched_counts(rank_thresh=1000):
    words = _load_mscoco_dict()
    top_ans_d = load_top_answers()
    count = 0
    word_id2top_ans_id = {}
    for w_id, w in enumerate(words):
        if w in top_ans_d:
            count += 1
            rank = top_ans_d[w]
            if rank < rank_thresh:
                print('%d: %s' % (rank, w))
                word_id2top_ans_id[w_id] = rank
    print('%d matched words, %d valid' % (count, len(word_id2top_ans_id)))
    return word_id2top_ans_id


def test():
    word_id2top_ans_id = stat_matched_counts()
    d = load_hdf5('data/capt1k_std_mscoco_val.data')
    capt_bow = d['att_arr']
    image_ids = d['image_ids']
    top_ans_cands = {}
    for i, (image_id, bow) in enumerate(zip(image_ids, capt_bow)):
        print(i)
        token_ids = np.where(bow == 1)[0].tolist()
        tmp = []
        for t_id in token_ids:
            if t_id in word_id2top_ans_id:
                tmp.append(t_id)
        top_ans_cands[image_id] = tmp
    pdb.set_trace()


if __name__ == '__main__':
    test()



