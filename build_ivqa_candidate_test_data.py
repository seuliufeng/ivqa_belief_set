import numpy as np
from time import time
from util import load_json
from w2v_answer_encoder import MultiChoiceQuestionManger
from nltk.tokenize import word_tokenize
from extract_vqa_word2vec_coding import SentenceEncoder
import pdb


def process():
    w2v_ncoder = SentenceEncoder()
    # load gt and answer manager
    ctx = MultiChoiceQuestionManger(subset='val')
    # load candidates
    candidates = load_json('result/var_vaq_beam_VAQ-VAR_full_kptest.json')
    # load candidate scores
    score_list = load_json('result/var_vaq_beam_VAQ-VAR_full_kptest_oracle_dump.json')
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
            dataset[question_id][question] = (score, aug_id)
        else:
            dataset[question_id] = {question: (score, aug_id)}

    # get stat
    unk_image_ids = set(unk_image_ids)
    num_images = len(unk_image_ids)
    print('Find %d unique keys from %d images' % (len(dataset), num_images))
    print('%0.3f questions on average' % (len(dataset) / float(num_images)))

    # build tuple
    num_pairs = 0
    offset = 0
    image_ids, quest_ids, aug_quest_ids, question_w2v, answer_w2v, scores = [], [], [], [], [], []
    num_task = len(dataset)
    t = time()
    for _i, (quest_id, item) in enumerate(dataset.items()):
        if _i % 1000 == 0:
            print('processed: %d/%d (%0.2f sec./batch)' % (_i, num_task, time() - t))
            t = time()
        ans = ctx.get_gt_answer(quest_id)
        image_id = ctx.get_image_id(quest_id)
        assert (image_id == question_id2image_id[quest_id])

        ps = []
        for q, (s, aug_id) in item.items():
            ps.append(q)
            aug_quest_ids.append(aug_id)
            scores.append(s)
        _this_n = len(ps)

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
    image_ids = np.array(image_ids, dtype=np.int32)
    quest_ids = np.array(quest_ids, dtype=np.int32)
    scores = np.array(scores, dtype=np.float32)
    aug_quest_ids = np.array(aug_quest_ids, dtype=np.int64)
    answer_w2v = np.concatenate(answer_w2v, axis=0).astype(np.float32)
    question_w2v = np.concatenate(question_w2v, axis=0).astype(np.float32)
    from util import save_hdf5
    sv_file = 'result/cst_ranking_kptest.data'
    save_hdf5(sv_file, {'image_ids': image_ids,
                        'quest_ids': quest_ids,
                        'aug_quest_ids': aug_quest_ids,
                        'scores': scores,
                        'answer_w2v': answer_w2v,
                        'question_w2v': question_w2v})


if __name__ == '__main__':
    process()
