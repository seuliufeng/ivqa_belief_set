import os
import json
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from util import load_json
import numpy as np
# from nltk.tokenize import word_tokenize
from inference_utils import vocabulary
from w2v_answer_encoder import MultiChoiceQuestionManger


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
    # question_vocab = OrderedDict(sorted(question_vocab.items(), key=lambda t: t[1], reverse=True))
    # import pdb
    # pdb.set_trace()
    new_vocab = {' '.join(_tokenize_sentence(k)): v for k, v in question_vocab.items()}
    return new_vocab


class MultipleChoiceEvaluater(object):
    def __init__(self, subset='val', num_eval=None, need_im_feat=True,
                 need_attr=False, use_ans_type=False, feat_type='res152'):
        anno_file = '../iccv_vaq/data/MultipleChoicesQuestionsKarpathy%sV2.0.json' % subset.title()
        self._subset = subset
        d = load_json(anno_file)
        self._id2type = d['candidate_types']
        self._annotations = d['annotation']
        if num_eval == 0:
            num_eval = len(self._annotations)
        self._num_to_eval = num_eval
        self._idx = 0
        self._need_attr = need_attr
        self._need_im_feat = need_im_feat
        self.num_samples = len(self._annotations)
            self._mc_ctx = MultiChoiceQuestionManger(subset='val')
        self._group_by_answer_type()
        self._use_ans_type = use_ans_type

    @property
    def annotations(self):
        return self._annotations

    def get_task_data(self):
        pass

    def get_labels(self, answer_ids):
        answer_id2labels = {info['answer_id']: info['labels'] for info in self._annotations}
        type_mat = []
        for ans_id in answer_ids:
            labels = np.array(answer_id2labels[ans_id])
            type_mat.append(labels[np.newaxis, :])
        type_mat = np.concatenate(type_mat, axis=0)
        return (type_mat == 0).argmax(axis=1)

    def _group_by_answer_type(self):
        self.answer_ids_per_type = {}
        for info in self._annotations:
            for quest_id in info['coco_question_ids']:
                answer_id = info['answer_id']
                type_str = self._mc_ctx.get_answer_type(quest_id)
                self.answer_ids_per_type.setdefault(type_str, []).append(answer_id)

    @staticmethod
    def _get_intersect_table(pool, target):
        # create hashing table
        hash_tab = {k: 0 for k in target}
        return np.array([c in hash_tab for c in pool])

    def evaluate_results(self, answer_ids, scores, model_type=None):
        types, results = [], []
        # ALL
        cmc = self._evaluate_worker(answer_ids, scores, 'ALL')
        results.append(cmc)
        types.append('all')
        # per answer type
        for type in self.answer_ids_per_type.keys():
            target = np.array(self.answer_ids_per_type[type])
            sel_tab = self._get_intersect_table(answer_ids, target)
            cmc = self._evaluate_worker(answer_ids[sel_tab],
                                        scores[sel_tab, :], type)
            results.append(cmc)
            types.append(type)
        results = np.concatenate(results, axis=0)
        if model_type is not None:
            from scipy.io import savemat
            res_file = 'result/mc_%s_result.mat' % model_type.lower()
            savemat(res_file, {'cmc': results, 'types': types})

    def _evaluate_worker(self, answer_ids, scores, type):
        answer_id2labels = {info['answer_id']: info['labels'] for info in self._annotations}
        type_mat = []
        for ans_id in answer_ids:
            labels = np.array(answer_id2labels[ans_id])
            type_mat.append(labels[np.newaxis, :])
        type_mat = np.concatenate(type_mat, axis=0)
        gt_mask = np.equal(type_mat, 0)

        gt_scores = []
        for i, (gt, score) in enumerate(zip(gt_mask, scores)):
            gt_scores.append(score[gt].max())
        # find the rank of gt scores
        gt_scores = np.array(gt_scores)[:, np.newaxis]
        sorted_scores = -np.sort(-scores, axis=1)
        gt_rank = np.equal(sorted_scores, gt_scores).argmax(axis=1)
        # print('\nMean rank: %0.2f' % gt_rank.mean())
        # compute cmc
        num, num_cands = gt_mask.shape
        cmc = np.zeros(num_cands, dtype=np.float32)
        for i in range(num_cands):
            cmc[i] = np.less_equal(gt_rank, i).sum()
        cmc = cmc / num * 100.
        print('\n=======   type %s  =======' % type.upper())
        print('----------  cmc   -----------')
        print('Top 1: %0.3f' % cmc[0])
        print('Top 3: %0.3f' % cmc[2])
        print('Top 10: %0.3f' % cmc[9])
        # top 1 analysis
        self.top1_analysis(scores, type_mat)
        return cmc[np.newaxis, :]

    def top1_analysis(self, scores, type_mat):
        # print('=======  Top 1 analysis  =======')
        print('---------  top 1  -----------')
        pred_labels = scores.argmax(axis=1)
        types = np.zeros_like(pred_labels)
        for i, idx in enumerate(pred_labels):
            types[i] = type_mat[i, idx]
        bin_count = np.bincount(types)
        num = pred_labels.size
        for i, c in enumerate(bin_count):
            type_str = self._id2type[str(i)]
            pnt = float(c) * 100. / num
            print('%s: %02.2f' % (type_str, pnt))
        print('\n')

    def prediction_examples(self):
        pass


def main():
    # get frequency
    qfreq = get_popular_questions('train')
    # get data
    ctx = MultipleChoiceEvaluater(subset='test')
    annos = ctx.annotations
    answer_ids, scores = [], []
    for item in annos:
        tmp = []
        answer_ids.append(item['answer_id'])
        for c in item['questions']:
            v = qfreq[c] if c in qfreq else 0
            tmp.append(v)
        scores.append(tmp)
    answer_ids = np.array(answer_ids, dtype=np.int32)
    scores = np.array(scores, dtype=np.float32)
    import pdb
    pdb.set_trace()
    ctx.evaluate_results(answer_ids, scores, 'qprior')


if __name__ == '__main__':
    main()


