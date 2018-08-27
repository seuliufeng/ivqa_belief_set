import numpy as np
from time import time
from util import load_json
from w2v_answer_encoder import MultiChoiceQuestionManger
from nltk.tokenize import word_tokenize
from extract_vqa_word2vec_coding import SentenceEncoder
import pdb


def process(delta = 0.2):
    # w2v_ncoder = SentenceEncoder()
    # load gt and answer manager
    ctx = MultiChoiceQuestionManger(subset='val')
    # load candidates
    candidates = load_json('result/var_vaq_beam_VAQ-VARDSDC_full.json')
    # load candidate scores
    score_list = load_json('result/var_vaq_beam_VAQ-VARDSDC_full_oracle_dump.json')
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

    # visualise
    vis_keys = dataset.keys()
    np.random.shuffle(vis_keys)

    for quest_id in vis_keys[:20]:
        ans = ctx.get_gt_answer(quest_id)
        image_id = ctx.get_image_id(quest_id)
        gt = ctx.get_question(quest_id).lower()
        print('\ngt: %s' % gt)
        for quest, sc in dataset[quest_id].items():
            print('%s (%0.3f)' % (quest, sc))



if __name__ == '__main__':
    process(delta=3.0)
