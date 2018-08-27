import numpy as np
from time import time
from util import load_json
from w2v_answer_encoder import MultiChoiceQuestionManger
from nltk.tokenize import word_tokenize
from extract_vqa_word2vec_coding import SentenceEncoder
import pdb


def process(delta = 0.2):
    w2v_ncoder = SentenceEncoder()
    # load gt and answer manager
    ctx = MultiChoiceQuestionManger(subset='train')
    # load candidates
    candidates = load_json('result/var_vaq_beam_VAQ-VAR_full_kptrain.json')
    # load candidate scores
    score_list = load_json('result/var_vaq_beam_VAQ-VAR_full_kptrain_oracle_dump.json')
    score_d = {item['aug_quest_id']: item['CIDEr'] for item in score_list}

    # loop over questions
    dataset = {}
    unk_image_ids = []
    question_id2image_id = {}
    for item in candidates:
        aug_id = item['question_id']
        question = item['question']
        image_id = item['image_id']
        unk_image_ids.append(image_id)
        question_id = int(aug_id / 1000)
        score = score_d[aug_id]
        question_id2image_id[question_id] = image_id
        if question_id in dataset:
            assert (question not in dataset[question_id])
            dataset[question_id][question] = score
        else:
            dataset[question_id] = {question: score}

    # get stat
    unk_image_ids = set(unk_image_ids)
    num_images = len(unk_image_ids)
    print('Find %d unique keys from %d images' % (len(dataset), num_images))
    print('%0.3f questions on average' % (len(dataset) / float(num_images)))

    # build tuple
    num_pairs = 0
    offset = 0
    cst_pairs = []
    image_ids, quest_ids, question_w2v, answer_w2v = [], [], [], []
    num_task = len(dataset)
    t = time()
    for _i, (quest_id, item) in enumerate(dataset.items()):
        if _i % 1000 == 0:
            print('processed: %d/%d (%0.2f sec./batch)' % (_i, num_task, time()-t))
            t = time()
        ans = ctx.get_gt_answer(quest_id)
        image_id = ctx.get_image_id(quest_id)
        assert(image_id == question_id2image_id[quest_id])

        gt = ctx.get_question(quest_id).lower()
        gt = ' '.join(word_tokenize(gt))
        include_gt = np.any(np.array(item.values()) == 10.)
        sc, ps = [], []
        if gt not in item and not include_gt:
            item[gt] = 10.
        for q, s in item.items():
            sc.append(s)
            ps.append(q)
        sc = np.array(sc, dtype=np.float32)
        _this_n = len(ps)
        path_ind = np.arange(_this_n) + offset
        # data checking and assertion
        try:
            assert (np.sum(sc == 10.) <= 1)  # only one gt
        except Exception as e:
            ind = np.where(sc == 10.)[0]
            for _idx in ind:
                print('%s' % (ps[_idx]))
            raise e

        # find contrastive pairs
        diff = sc[np.newaxis, :] - sc[:, np.newaxis]
        valid_entries = diff >= delta
        neg, pos = np.where(valid_entries)
        assert (np.all(np.greater_equal(sc[pos] - sc[neg], delta)))
        pos_q_ind = path_ind[pos]
        neg_q_ind = path_ind[neg]

        # save
        _this_pairs = [[p, n] for p, n in zip(pos_q_ind, neg_q_ind)]
        cst_pairs += _this_pairs

        # encode answer
        _ans_w2v = w2v_ncoder.encode(ans)
        ans_w2v = np.tile(_ans_w2v, [_this_n, 1])
        answer_w2v.append(ans_w2v)

        # encode questions
        for p in ps:
            _q_w2v = w2v_ncoder.encode(p)
            question_w2v.append(_q_w2v)
            image_ids.append(image_id)
            quest_ids.append(quest_id)

        # update pointer
        offset += _this_n
        num_pairs += _this_n

    print('Total pairs: %d' % num_pairs)

    # merge
    cst_pairs = np.array(cst_pairs, dtype=np.int32)
    image_ids = np.array(image_ids, dtype=np.int32)
    quest_ids = np.array(quest_ids, dtype=np.int32)
    answer_w2v = np.concatenate(answer_w2v, axis=0).astype(np.float32)
    question_w2v = np.concatenate(question_w2v, axis=0).astype(np.float32)
    from util import save_hdf5
    sv_file = 'result/cst_ranking_kptrain_delta%g.data' % delta
    save_hdf5(sv_file, {'cst_pairs': cst_pairs,
                        'image_ids': image_ids,
                        'quest_ids': quest_ids,
                        'answer_w2v': answer_w2v,
                        'question_w2v': question_w2v})


if __name__ == '__main__':
    process(delta=3.0)
