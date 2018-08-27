import os
import json
from collections import OrderedDict
from nltk.tokenize import word_tokenize


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


def get_popular_questions(subset):
    print('Loading annotation files [%s]...' % subset)
    quest_file = 'data/annotations/MultipleChoice_mscoco_%s2014_questions.json' % subset
    d = json.load(open(quest_file, 'r'))
    questions = d['questions']
    print('Tokenize candidate answers...')
    question_vocab = {}
    for info in questions:
        quest = info['question']
        quest_key = quest.lower()
        if quest_key in question_vocab:
            question_vocab[quest_key] += 1
        else:
            question_vocab[quest_key] = 1
    # sort keys
    question_vocab = OrderedDict(sorted(question_vocab.items(), key=lambda t: t[1], reverse=True))
    # import pdb
    # pdb.set_trace()
    return ' '.join(_tokenize_sentence(question_vocab.items()[0][0]))


def make_dummpy_result_file(pop_q):
    from util import load_json, save_json
    results = load_json('result/quest_vaq_greedy_VAQ-SAT_kptest.json')
    for item in results:
        item['image_id'] = int(item['image_id'])
        item['question_id'] = int(item['question_id'])
        item['question'] = pop_q
    sv_file = 'result/tmp_pop_q_kptest.json'
    save_json(sv_file, results)
    return sv_file


def evaluate_question(result_file, subset='kpval', version='v1'):
    from analysis.eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()

    subset = 'train' if subset == 'train' else 'val'
    if version == 'v1':
        annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
        question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)
    elif version == 'v2':
        anno_dir = '../../data/VQA2.0'
        annotation_file = '%s/v2_mscoco_%s2014_annotations.json' % (anno_dir, subset)
        question_file = '%s/v2_OpenEnded_mscoco_%s2014_questions.json' % (anno_dir, subset)
    else:
        raise Exception('unknown version, v1 or v2')

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    # return evaluator.get_overall_blue4()
    return evaluator.get_overall_cider()


if __name__ == '__main__':
    pop_q = get_popular_questions('train')
    print('Most popular questions: %s' % pop_q)
    res_file = make_dummpy_result_file(pop_q)
    evaluate_question(res_file, 'kptest')
