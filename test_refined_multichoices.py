from __future__ import division
import tensorflow as tf

from util import get_res5c_feature_root, load_hdf5, load_json
import numpy as np
from w2v_answer_encoder import MultiChoiceQuestionManger

tf.logging.set_verbosity(tf.logging.INFO)


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


class MultipleChoiceEvaluater(object):
    def __init__(self, subset='val'):
        anno_file = 'data/MultipleChoicesQuestionsKarpathy%s.json' % subset.title()
        self._subset = subset
        d = load_json(anno_file)
        self._id2type = d['candidate_types']
        self._annotations = d['annotation']
        self._idx = 0
        self.num_samples = len(self._annotations)
        self._mc_ctx = MultiChoiceQuestionManger(subset='val')
        self._group_by_answer_type()

    def update_annotation(self, do_update=True):
        man_file = 'data/distractor_analysis.json'
        anno = load_json(man_file)['annotation']
        hash_tab = {d['answer_idx']: d['confused'] for d in anno}
        if not do_update:
            return np.array([d['answer_idx'] for d in anno if d['confused']])
            # return np.array(hash_tab.keys())

        for datum in self._annotations:
            ans_id = datum['answer_id']
            datum['labels'] = np.array(datum['labels'])
            if ans_id in hash_tab:
                conf_ids = hash_tab[ans_id]
                if conf_ids:
                    tmp_ids = np.array(conf_ids)
                    datum['labels'][tmp_ids] = 0
        return np.array(hash_tab.keys())

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
        print('Top 5: %0.3f' % cmc[4])
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


def vaq_multiple_choices():
    refine = False
    pred_path = '/import/vision-datasets001/fl302/code/iccv_vaq/result/VAQ-A_mc_prediction.h5'
    # pred_path = '/import/vision-datasets001/fl302/code/iccv_vaq/result/VAQ-lstm-dec-sup_mc_prediction.h5'
    # set evaluator
    mc_ctx = MultipleChoiceEvaluater(subset='val')
    v_ans_ids = mc_ctx.update_annotation(refine)

    # load predictions
    d = load_hdf5(pred_path)
    answer_ids = d['answer_ids']
    predictions = d['scores']
    aid2index = {aid: i for (i, aid) in enumerate(answer_ids)}
    slice_index = np.array([aid2index[id] for id in v_ans_ids])
    sliced_preds = predictions[slice_index]
    # evaluate
    mc_ctx.evaluate_results(v_ans_ids, sliced_preds,
                            model_type='NN')


if __name__ == '__main__':
    vaq_multiple_choices()
