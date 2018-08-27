try:
    from gensim.models import Word2Vec
except:
    pass
import tensorflow as tf
import numpy as np
import re
from util import pickle, unpickle

EPS = 1e-8


class Word2VecEncoder(object):
    def __init__(self, word2vec_file):
        d = unpickle(word2vec_file)
        self._vocab = d['vocab']
        self._word_vectors = d['word_vectors']
        self._create_index_dict(d['word2idx'])

    def _create_index_dict(self, index):
        index = np.array(index)
        dig2idx = index[-10:].tolist()
        index[-10:] = 0
        self._num2idx = {str(k): dig2idx[k] for k in range(10)}
        self._word2idx = {k: v for (k, v) in zip(self._vocab, index.tolist())}

    def encode(self, sentence):
        if sentence is None:
            raise Exception('input can'' be none')
        digits = ''.join(re.findall(r'\d+', ''.join(sentence)))
        index = []
        if digits:  # if there are digits
            for i in list(digits):
                index.append(self._num2idx[i])
        for word in sentence:
            try:
                index.append(self._word2idx[word])
            except Exception, e:
                pass
                # print('Warning: word %s not in dictionary' % word)
        v = self._word_vectors[index].sum(axis=0)
        index = np.array(index)
        n = (index > 0).sum() + EPS
        return v / n


def load_vocabulary(vocab_file):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    return [line.split()[0] for line in reverse_vocab]


def get_combined_vocabulary():
    vocab_quest = load_vocabulary('data/vqa_all_question_word_counts.txt')
    vocab_ans = load_vocabulary('data/vqa_all_answer_word_counts.txt')
    vocab = {}
    for word in vocab_quest + vocab_ans:
        vocab[word] = vocab.get(word, 0) + 1
    vocab = vocab.keys()
    # add digits
    digit_vocab = [str(i) for i in range(10)]
    return vocab + digit_vocab


def slice_word2vec_model(vocab):
    from time import time
    t = time()
    print('loading word2vec model')
    model = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
    print('model loaded (%0.2f sec.)' % (time() - t))
    word_vec_dim = model['cat'].size
    word_vectors = []
    word_to_idx = []

    idx = 1
    word_vectors.append(np.zeros([1, word_vec_dim], np.float32))
    for word in vocab:
        try:
            word_vectors.append(model[word].reshape([1, -1]))
            word_to_idx.append(idx)
            idx += 1
        except Exception, e:
            word_to_idx.append(0)
    word_vectors = np.concatenate(word_vectors, axis=0)
    num = word_vectors.shape[0]
    assert (num == idx)
    assert (len(word_to_idx) == len(vocab))
    print('%d words in corpus' % idx)
    pickle('data/vqa_word2vec_model.pkl', {'vocab': vocab,
                                           'word2idx': word_to_idx,
                                           'word_vectors': word_vectors})


if __name__ == '__main__':
    # vocab = get_combined_vocabulary()
    # slice_word2vec_model(vocab)
    encoder = Word2VecEncoder('data/vqa_word2vec_model.pkl')


    def debug_encoder_once(s):
        print s
        print encoder.encode(s)


    debug_encoder_once(['<S>', '12', '</S>'])
    debug_encoder_once(['<S>', 'red', 'and', 'gray', '</S>'])
    debug_encoder_once(None)
