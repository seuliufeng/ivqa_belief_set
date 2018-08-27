import tensorflow as tf
from time import time
import os
from collections import namedtuple
from util import load_json
import itertools
import pdb

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "question_id", "question", "answer",
                            "question_type", "answer_type"])


def load_and_process_metadata(subset):
    tf.logging.info('Processing meta data of %s...' % subset)
    t = time()
    is_test = subset.startswith('test')
    year = 2015 if is_test else 2014
    subtype = '%s%d' % (subset, year)
    ann_root = '/usr/data/fl302/code/VQA-tensorflow/data/annotations'
    datatype = 'test2015' if is_test else subtype
    # tf.logging.info('Loading annotations and questions...')
    questions = load_json(os.path.join(ann_root, 'MultipleChoice_mscoco_%s_questions.json' % subtype))['questions']
    dataset = questions if is_test \
        else load_json(os.path.join(ann_root, 'mscoco_%s_annotations.json' % subtype))['annotations']
    # pdb.set_trace()

    meta = []
    for info, quest in zip(dataset, questions):
        ans = None if is_test else info['multiple_choice_answer']
        # token_ans = None if is_test else ans
        quest_id = info['question_id']
        image_id = info['image_id']
        question = quest['question']
        # mc_ans = quest['multiple_choices']
        question_type = info['question_type']
        answer_type = info['answer_type']
        meta.append(ImageMetadata(image_id, None, quest_id, question, ans,
                                  question_type, answer_type))
    tf.logging.info('Time %0.2f sec.' % (time() - t))
    return meta


def load_top_answer_vocab():
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    answer_vocab = []
    with open(top_ans_file, 'r') as fs:
        for line in fs:
            answer_vocab.append(line.strip())
    answer_vocab = {ans: i for i, ans in enumerate(answer_vocab)}
    return answer_vocab


def parse_by_types(top_ans_vocab):
    import nltk
    type_dict = {}
    verbs = ['VB', 'VBG', 'VBP', 'VBZ']
    verb_hash_t = {v: {} for v in verbs}
    attributes = ['RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS', 'VBN', 'VBD']
    attr_hash_t = {v: {} for v in attributes}
    objects = ['NN', 'NNS', 'NNP', 'NNPS']
    object_hash_t = {v: {} for v in objects}
    for i, key in enumerate(top_ans_vocab):
        parsed = nltk.pos_tag(nltk.word_tokenize(key))
        # only keep all consistent
        if len(parsed) > 1:
            tmp_d = {}
            for (p, t) in parsed:
                tmp_d.update({t: p})
            if len(tmp_d) > 1:
                continue

        for (p, t) in parsed:
            if t in type_dict:
                type_dict[t][key] = i
            else:
                type_dict[t] = {key: i}
    # sort to different types
    verb_dict, attr_dict, object_dict = {}, {}, {}
    for t in type_dict:
        if t in verb_hash_t:
            verb_dict.update(type_dict[t])
        if t in attr_hash_t:
            attr_dict.update(type_dict[t])
        if t in object_hash_t:
            object_dict.update(type_dict[t])
    return {'verb': verb_dict,
            'attr': attr_dict,
            'noun': object_dict}


def get_stat():
    meta = load_and_process_metadata('val')
    top_ans_vocab = load_top_answer_vocab()
    ptb_dict = parse_by_types(top_ans_vocab)
    for k, v in ptb_dict.iteritems():
        print('type %s: %d' % (k, len(v)))

    question_type_vocab = {}
    answer_type_vocab = {}
    for info in meta:
        answer = info.answer
        if answer not in top_ans_vocab:
            continue
        # update question vocab
        qtype = info.question_type
        if qtype in question_type_vocab:
            if answer not in question_type_vocab[qtype]:
                question_type_vocab[qtype][answer] = None
        else:
            question_type_vocab[qtype] = {answer: None}

        atype = info.answer_type
        if atype in answer_type_vocab:
            if answer not in answer_type_vocab[atype]:
                answer_type_vocab[atype][answer] = None
        else:
            answer_type_vocab[atype] = {answer: None}

    print('Number of question types: %d' % len(question_type_vocab))
    print('Unique answers for each question type: ')
    for qtype in question_type_vocab:
        print('%s: %d' % (qtype, len(question_type_vocab[qtype])))

    print('\nNumber of answer types: %d' % len(answer_type_vocab))
    print('Unique answers for each answer type: ')
    for atype in answer_type_vocab:
        print('%s: %d' % (atype, len(answer_type_vocab[atype])))
    merge_answer(question_type_vocab, ptb_dict, top_ans_vocab)


def merge_answer(question_vocab, ptb_vocab, top_ans_vocab):
    keywords = {'color': 'color',
                'sport': 'sport',
                'brand': 'brand',
                'time': 'when',
                'where is the': 'where',
                'why': 'why',
                'what': 'what',
                'animal': 'animal',
                'how many': 'counts',
                'room': 'room',
                'what is on the': 'object',
                'what is the name': 'object',
                }
    stat_vocab = {t: {} for t in keywords.values()}
    for key in question_vocab:
        for pattern in keywords:
            if pattern in key:
                ptype = keywords[pattern]
                stat_vocab[ptype].update(question_vocab[key])
    # add type other nouns
    black_list = {}
    b_keys = {'where', 'when', 'counts', 'why'}
    for k in b_keys:
        black_list.update(stat_vocab[k])
    what = stat_vocab['what']
    new_what = {}
    for k in what:
        if k not in black_list:
            new_what[k] = what[k]
    stat_vocab['what'] = new_what

    # other_nouns = []
    # for n in ptb_vocab['noun'].keys():
    #     if n in black_list:
    #         continue
    #     other_nouns.append(n)
    # stat_vocab['others_noun'] = other_nouns

    includes = []
    for v in stat_vocab.values():
        includes += v
    top_1k_inds = [top_ans_vocab[v] for v in includes if top_ans_vocab[v] < 1000]
    top_500_inds = [top_ans_vocab[v] for v in includes if top_ans_vocab[v] < 500]
    import numpy as np
    missed = np.setdiff1d(range(500), top_500_inds)
    inverse_d = {i: k for k, i in top_ans_vocab.iteritems()}
    print('\nMissed top answers:')
    for missed_id in missed:
        print('%s' % inverse_d[missed_id])

    print('\nIncludes %d top 500 answers' % len(top_500_inds))
    print('Includes %d top 1000 answers' % len(top_1k_inds))
    print('Includes %d top 2000 answers' % len(includes))

    print('\nNumber of organised types: %d' % len(stat_vocab))
    print('Unique answers for each type: ')
    for atype in stat_vocab:
        print('%s: %d' % (atype, len(stat_vocab[atype])))
    pdb.set_trace()


def intersect_with_coco_attribute():
    from sklearn.externals import joblib
    cocottributes = joblib.load('/usr/data/fl302/data/coco_attr/cocottributes_eccv_version.jbl')
    attr_names = [item['name'] for item in cocottributes['attributes']]
    names = []
    for attr in attr_names:
        atoms = attr.split('/')
        for a in atoms:
            names.append(a.strip())
    attr_names = names
    top_ans_vocab = load_top_answer_vocab()
    matched = 0
    for name in attr_names:
        if name in top_ans_vocab:
            matched += 1
    print('Matched %d top answers' % matched)
    return attr_names


def load_vocab(fpath, reverse=False):
    from collections import OrderedDict
    vocab = []
    with open(fpath, 'r') as fs:
        for line in fs:
            word = line.split(' ')[0]
            vocab.append(word)
    if reverse:
        return vocab
    vocab_dict = OrderedDict()
    for i, word in enumerate(vocab):
        vocab_dict[word] = i
    return vocab_dict


def dict_intersect_with_coco_attribute(attr_names):
    ans_voc_file = 'data/vqa_trainval_answer_word_counts.txt'
    ans_word_dict = load_vocab(ans_voc_file)
    matched = 0
    for name in attr_names:
        if name in ans_word_dict:
            matched += 1
    print('Matched %d words in answers' % matched)
    return attr_names


if __name__ == '__main__':
    attr_names = intersect_with_coco_attribute()
    dict_intersect_with_coco_attribute(attr_names)
    # get_stat()
