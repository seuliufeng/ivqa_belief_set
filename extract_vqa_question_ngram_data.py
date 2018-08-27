from collections import namedtuple
import tensorflow as tf
from time import time
import os
import re
import numpy as np
from util import save_json, load_json, save_hdf5
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from collections import defaultdict
import pdb

EPS = 1e-8
tf.logging.set_verbosity(tf.logging.INFO)


class SentenceEncoder(object):
    def __init__(self, average=True):
        vocab_file = '/usr/data/fl302/code/VQA-tensorflow/data/GoogleNews-vectors-negative300.bin'
        t = time()
        tf.logging.info('Loading Word2Vec models...')
        self.vocab = KeyedVectors.load_word2vec_format(vocab_file, binary=True)
        tf.logging.info('Model loaded, time %0.2f' % (time() - t))
        self.average = average

    def encode(self, sentence):
        tokens = _tokenize_sentence(sentence)
        vec = np.zeros((300,), dtype=np.float32)
        # print(tokens)
        for t in tokens:
            if t in self.vocab:
                tmp = self.vocab[t]
            else:
                tmp = self.vocab['UNK']
            vec = vec + tmp
        if self.average:
            vec = vec / (len(tokens) + EPS)
        return vec[np.newaxis, :]


ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "question_id", "question", "answer",
                            "token_answer", "choices"])


def _tokenize_sentence(sentence, replace=False):
    # replace ' with word pattern
    sentence = sentence.lower()
    if replace:
        apostrophe_pat = 'apostrophe'
        sentence = sentence.replace("'", apostrophe_pat)
        regex = r'\w+'
        tokens = re.findall(regex, sentence)
        new_tokens = []
        for t in tokens:
            t = t.replace(apostrophe_pat, "'")
            new_tokens.append(t)
        return new_tokens
    else:
        regex = r'\w+'
        tokens = re.findall(regex, sentence)
        return tokens
        # return word_tokenize(str(sentence).lower())


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

    meta = []
    for info, quest in zip(dataset, questions):
        ans = None if is_test else info['multiple_choice_answer']
        token_ans = None if is_test else ans
        quest_id = info['question_id']
        image_id = info['image_id']
        question = quest['question']
        mc_ans = quest['multiple_choices']
        meta.append(ImageMetadata(image_id, None, quest_id, question, ans, token_ans, mc_ans))
    tf.logging.info('Time %0.2f sec.' % (time() - t))
    return meta


def _encode_answer_candidates(info, encoder):
    gt = info.answer
    labels = []
    cand_coding = []
    for cand in info.choices:
        labels.append(gt == cand)
        cand_coding.append(encoder.encode(cand))
    cand_coding = np.concatenate(cand_coding, axis=1)
    label = np.array(labels).argmax()
    assert(sum(labels) == 1)
    return cand_coding, label


def _encode_w2v(images, encoder, subset):
    quest_coding = []
    cands_coding = []
    labels = []
    quest_ids = []
    cands_meta = []
    for i, info in enumerate(images):
        if not i % 1000:
            tf.logging.info("%s: processed %d of %d items." % (subset.upper(), i, len(images)))

        quest_id = info.question_id
        q_w2v = encoder.encode(info.question)
        ca_w2v, label = _encode_answer_candidates(info, encoder)
        # pdb.set_trace()
        quest_coding.append(q_w2v)
        cands_coding.append(ca_w2v)
        labels.append(label)
        quest_ids.append(quest_id)
        _m = {'quest_id': quest_id, 'cands': info.choices}
        cands_meta.append(_m)
    # ready to pack data
    quest_coding = np.concatenate(quest_coding, axis=0).astype(np.float32)
    cands_coding = np.concatenate(cands_coding, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)
    quest_ids = np.array(quest_ids, dtype=np.int32)
    save_hdf5('data3/vqa_mc_w2v_coding_%s.data' % subset, {'quest_w2v': quest_coding,
                                                           'cands_w2v': cands_coding,
                                                           'labels': labels,
                                                           'quest_ids': quest_ids})
    save_json('data3/vqa_mc_cands_%s.meta' % subset, cands_meta)


def split_data_by_seed(images, subset):
    d = load_json('data/vqa_std_mscoco_%s.meta' % subset)
    quest_ids = d['quest_id']
    # get quest_id to index mappling
    qid2index = {info.question_id: i for (i, info) in enumerate(images)}

    new_images = []
    for quest_id in quest_ids:
        idx = qid2index[quest_id]
        info = images[idx]
        new_images.append(info)
        assert (info.question_id == quest_id)
    return new_images


def _process_subset(meta, encoder, subset):
    images = split_data_by_seed(meta, subset)
    _encode_w2v(images, encoder, subset)


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def build_n_gram_vocab(meta, n=4):
    n_gram_dict = {}
    for info in meta:
        q = info.question
        tokens = _tokenize_sentence(q)
        q_len = len(tokens)
        tmp_dict = {}
        for i in range(0, q_len - 3):
            n_gram = ' '.join(tokens[i: i + n])
            tmp_dict[n_gram] = 1
        n_gram_dict.update(tmp_dict)
    return n_gram_dict


def _process():
    train = load_and_process_metadata('train')
    val = load_and_process_metadata('val')
    encoder = SentenceEncoder()
    meta = train + val
    _process_subset(meta, encoder, 'kptrain')
    _process_subset(meta, encoder, 'kpval')
    _process_subset(meta, encoder, 'kptest')
    _process_subset(meta, encoder, 'kprestval')


if __name__ == '__main__':
    _process()
