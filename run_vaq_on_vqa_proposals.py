from __future__ import division
import tensorflow as tf

import os
import numpy as np
from util import save_hdf5, update_progress, load_hdf5
from vqa_model_creater import get_model_creation_fn
from config import ModelConfig
from mtl_data_fetcher import AttentionTestDataFetcher as Reader
from inference_utils.question_generator_util import SentenceGenerator
from inference_utils import vocabulary
from nltk.tokenize import word_tokenize
import json

tf.flags.DEFINE_string("model_type", "VAQ-lstm-dec-sup",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/kpvaq_%s",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


def _load_reverse_answer_vocab(vocab_file='data/vqa_trainval_top2000_answers.txt'):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    reverse_vocab = [line.strip() for line in reverse_vocab]
    return reverse_vocab


class AnswerProposals(object):
    def __init__(self, top_k):
        prop_file = 'data/vqa-st_vqa_score2000_kpval.hdf5'
        d = load_hdf5(prop_file)
        quest_ids = d['quest_ids']
        self.quest_id2score_index = {int(qid): i for i, qid in enumerate(quest_ids)}
        self.scores = d['confidence'][:, :2000]
        self.top_k = top_k
        self.encoder = SentenceEncoder('answer')
        self.reverse_top_ans_dict = _load_reverse_answer_vocab()

    def answer_ids_to_sequences(self, answer_ids):
        # for ans_id in answer_ids:
        answers = [self.reverse_top_ans_dict[a_id] for a_id in answer_ids]
        return self.encoder.encode_sentences(answers)

    def get_answer(self, quest_id):
        idx = self.quest_id2score_index[quest_id]
        # slice
        score = self.scores[idx].copy()
        answer_ids = (-score).argsort()[:self.top_k]
        scores = score[answer_ids]
        ans_arr, ans_len = self.answer_ids_to_sequences(answer_ids)
        return ans_arr, ans_len, scores, answer_ids


class SentenceEncoder(object):
    def __init__(self, type='question'):
        assert (type in ['answer', 'question'])
        self._vocab = None
        vocab_file = 'data/vqa_%s_%s_word_counts.txt' % ('trainval', type)
        self._load_vocabulary(vocab_file)

    def _load_vocabulary(self, vocab_file=None):
        print('Loading answer words vocabulary...')
        print(vocab_file)
        self._vocab = vocabulary.Vocabulary(vocab_file)

    def encode_sentences(self, sentences):
        exist_multiple = type(sentences) is list
        if exist_multiple:
            tokens_list = [self._encode_sentence(s) for s in sentences]
            return self._put_to_numpy_array(tokens_list)
        else:
            tokens = np.array(self._encode_sentence(sentences),
                              dtype=np.int32)
            token_len = np.array(tokens.size, dtype=np.int32)
            return tokens.flatten(), token_len.flatten()

    def _encode_sentence(self, sentence):
        tokens = _tokenize_sentence(sentence)
        return [self._vocab.word_to_id(word) for word in tokens]

    @staticmethod
    def _put_to_numpy_array(tokens):
        seq_len = [len(q) for q in tokens]
        max_len = max(seq_len)
        num = len(tokens)
        token_arr = np.zeros([num, max_len], dtype=np.int32)
        for i, x in enumerate(token_arr):
            x[:seq_len[i]] = tokens[i]
        seq_len = np.array(seq_len, dtype=np.int32)
        return token_arr, seq_len


def pre_process_input_data(inputs, ans_prop):
    im_feed, attr_feed, quest, quest_len, _, quest_id, image_id = inputs
    im_feed = im_feed.reshape([1, 14, 14, 2048])
    attr_feed = attr_feed.reshape([1, 1000])
    quest = quest.reshape([1, -1])

    # convert type
    quest_id = int(quest_id)
    ans_arr, ans_len, scores, answer_ids = ans_prop.get_answer(quest_id)
    # replicate
    num = ans_arr.shape[0]
    im_feed = np.tile(im_feed, [num, 1, 1, 1])
    attr_feed = np.tile(attr_feed, [num, 1])
    quest = np.tile(quest, [num, 1])
    quest_len = np.tile(quest_len, num)
    feed_inputs = [im_feed, attr_feed, quest, quest_len, None, ans_arr, ans_len]
    return feed_inputs, scores, answer_ids


def vaq_condition(checkpoint_path=None):
    subset = 'kpval'
    model_config = ModelConfig()
    model_config.cell_option = 4

    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = Reader(batch_size=1, subset=subset, output_attr=True, output_im=True,
                    output_qa=True, output_capt=False, output_ans_seq=False)
    ans_prop_ctx = AnswerProposals(top_k=5)

    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'evaluate')
        model.build()
        saver = tf.train.Saver()

        sess = tf.Session()
        tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
        saver.restore(sess, checkpoint_path)

    fetch_op = model.losses
    num_batches = reader.num_batches

    save_file = 'data/%s_vqa_vaq_rerank_%s.hdf5' % ((FLAGS.model_type).lower(), subset)
    print('Save File: %s' % save_file)
    print('Running conditioning...')
    vaq_scores, quest_ids = [], []
    vqa_scores, vqa_pred_labels = [], []
    for i in range(num_batches):
        update_progress(i / float(num_batches))

        outputs = reader.get_test_batch()
        im_feed, attr_feed, quest, _, _, quest_id, image_id = outputs
        quest_id = int(quest_id)
        outputs, vqa_score, answer_ids = pre_process_input_data(outputs, ans_prop_ctx)

        losses = sess.run(fetch_op, feed_dict=model.fill_feed_dict(outputs))
        vaq_score = losses[:, :-1].mean(axis=1)
        vaq_score = vaq_score[np.newaxis, ::]
        vaq_scores.append(vaq_score)
        quest_ids.append(quest_id)
        vqa_scores.append(vqa_score[np.newaxis, :])
        vqa_pred_labels.append(answer_ids[np.newaxis, :])

    vaq_scores = np.concatenate(vaq_scores, axis=0)
    vqa_scores = np.concatenate(vqa_scores, axis=0)
    vqa_pred_labels = np.concatenate(vqa_pred_labels, axis=0)
    quest_ids = np.array(quest_ids, dtype=np.int32)
    print('\nSaving result files: %s...' % save_file)
    save_hdf5(save_file, {'vaq_scores': vaq_scores,
                          'vqa_scores': vqa_scores,
                          'vqa_pred_labels': vqa_pred_labels,
                          'quest_ids': quest_ids})


def score_fusion():
    subset = 'kpval'
    EPS = 1e-12
    T = 3.0
    save_file = 'data/%s_vqa_vaq_rerank_%s.hdf5' % ((FLAGS.model_type).lower(), subset)
    d = load_hdf5(save_file)
    quest_ids = d['quest_ids']
    vqa_scores = d['vqa_scores']
    vaq_scores = d['vaq_scores']
    vqa_pred_labels = d['vqa_pred_labels']

    # context
    to_sentence = SentenceGenerator(trainset='trainval')

    # fusion
    ans_ids = []
    for i, (quest_id, vqa_score, vaq_score, pred_label) in enumerate(zip(quest_ids,
                                                                         vqa_scores,
                                                                         vaq_scores,
                                                                         vqa_pred_labels)):
        vaq_score = np.exp(-T * vaq_score)
        vaq_score /= (vaq_score.sum() + EPS)
        score = vaq_score * vqa_score
        score = vqa_score
        idx = score.argmax()
        pred = pred_label[idx]
        # add this to result
        ans_ids.append(pred)

    result = [{u'answer': to_sentence.index_to_top_answer(aid),
               u'question_id': int(qid)} for aid, qid in zip(ans_ids, quest_ids)]

    # save results
    tf.logging.info('Saving results')
    res_file = 'vaq_on_vqa_proposal_tmp.json'
    json.dump(result, open(res_file, 'w'))
    tf.logging.info('Done!')
    tf.logging.info('#Num eval samples %d' % len(result))
    return res_file, quest_ids


def main(_):
    # vaq_condition()
    from vqa_eval import evaluate_model
    res_file, quest_ids = score_fusion()
    acc = evaluate_model(res_file, quest_ids)
    print(acc)


if __name__ == '__main__':
    tf.app.run()
