from __future__ import division
import tensorflow as tf

import os
from util import get_res5c_feature_root, load_hdf5, load_json
import numpy as np
from nltk.tokenize import word_tokenize
from inference_utils import vocabulary
from w2v_answer_encoder import MultiChoiceQuestionManger

tf.logging.set_verbosity(tf.logging.INFO)


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


def add_answer_type(quest_id, mc_ctx):
    answer_type_id = mc_ctx.get_answer_type_coding(quest_id)
    answer_type_id = np.array(answer_type_id, dtype=np.int32).reshape([1, ])
    return answer_type_id


def evaluate_question(result_file, subset='kptest'):
    from eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()
    # assert (subset in ['train', 'dev', 'val'])
    subset = 'train' if subset == 'train' else 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    return evaluator.get_overall_cider()


class MCDataFetcher(object):
    def __init__(self, subset='kpval', feat_type='res152'):
        self._subset = subset
        # load attribute file
        if feat_type != 'res152':
            data_file = 'data/attribute_std_mscoco_%s.data' % self._subset
            d = load_hdf5(data_file)
            self._attributes = d['att_arr'].astype(np.float32)
        else:
            data_file = 'data/res152_std_mscoco_%s.data' % self._subset
            d = load_hdf5(data_file)
            self._attributes = d['features'].astype(np.float32)
        image_ids = d['image_ids']
        self._image_id2index = {image_id: i for i, image_id in enumerate(image_ids)}

    def get_image_feature(self, image_id):
        self._im_feat_root = os.path.join(get_res5c_feature_root(),
                                          'val2014')
        filename = 'COCO_val2014_%012d.jpg' % image_id
        f = np.load(os.path.join(self._im_feat_root, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))

    def get_attribute_feature(self, image_id):
        index = self._image_id2index[image_id]
        return self._attributes[index]


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


class MultipleChoiceEvaluater(object):
    def __init__(self, subset='val', num_eval=None, need_im_feat=True,
                 need_attr=False, use_ans_type=False, feat_type='res152'):
        anno_file = '/import/vision-datasets001/fl302/code/iccv_vaq/data/MultipleChoicesQuestionsKarpathy%sV2.0.json' % subset.title()
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
        self._quest_encoder = SentenceEncoder('question')
        self._answer_encoder = SentenceEncoder('answer')
        self._im_encoder = MCDataFetcher(subset='kp%s' % subset,
                                         feat_type=feat_type)
        self.num_samples = len(self._annotations)
        self._mc_ctx = MultiChoiceQuestionManger(subset='val')
        self._group_by_answer_type()
        self._use_ans_type = use_ans_type

    def get_task_data(self):
        info = self._annotations[self._idx]
        questions = info['questions']
        answer = info['answer']
        answer_idx = info['answer_id']
        image_id = info['image_id']
        quest_id = int(info['coco_question_ids'][0])
        # prepare for output
        outputs = []
        if self._need_im_feat:
            im_feat = self._im_encoder.get_image_feature(image_id)
            outputs.append(im_feat)
        if self._need_attr:
            attr = self._im_encoder.get_attribute_feature(image_id)
            outputs.append(attr)
        quest, quest_len = self._quest_encoder.encode_sentences(questions)
        if self._use_ans_type:
            ans_type = add_answer_type(quest_id, self._mc_ctx)
            outputs += [quest, quest_len, None, ans_type, answer_idx, image_id]
        else:
            ans, ans_len = self._answer_encoder.encode_sentences(answer)
            outputs += [quest_id, quest, quest_len, None, ans, ans_len, answer_idx, image_id]
        self._idx += 1
        return outputs

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
        print('Top 1: %0.2f' % cmc[0])
        print('Top 5: %0.2f' % cmc[4])
        print('Top 10: %0.2f' % cmc[9])
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


def vaq_multiple_choices():
    subset = 'test'
    need_attr = False
    need_im_feat = False
    use_answer_type = False
    # feat_type = 'semantic'
    feat_type = 'res152'

    from test_nn_question_generator import NNModel

    mc_ctx = MultipleChoiceEvaluater(subset=subset,
                                     need_im_feat=need_im_feat,
                                     need_attr=need_attr,
                                     feat_type=feat_type,
                                     use_ans_type=use_answer_type)

    num_batches = mc_ctx.num_samples
    nn_model = NNModel(subset=subset)

    print('Running multiple choices...')
    predictions, answer_ids = [], []
    for i in range(num_batches):
        if i % 10 == 0:
            print('Running multiple choices: %d/%d' % (i, num_batches))
        outputs = mc_ctx.get_task_data()
        quest_id, capt, capt_len, _, _, _, ans_idx, image_id = outputs
        scores = nn_model.score_multi_choices(quest_id, capt, capt_len)
        predictions.append(scores[np.newaxis, :])
        answer_ids.append(ans_idx)

    predictions = np.concatenate(predictions, axis=0)
    answer_ids = np.array(answer_ids)

    # evaluate
    mc_ctx.evaluate_results(answer_ids, predictions,
                            model_type='NN')


if __name__ == '__main__':
    vaq_multiple_choices()
