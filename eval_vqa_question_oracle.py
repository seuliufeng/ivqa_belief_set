import json
import sys
import numpy as np
from util import get_dataset_root

vqa_data_root, toolkit_dir = get_dataset_root()
sys.path.insert(0, toolkit_dir)
# from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.coco_eval_toolkit_oracle import COCOEvalCap
import pdb
from time import time

#
# def get_dataset_root():
#     import os
#     if os.path.exists('/import/vision-ephemeral/fl302/data/VQA'):
#         data_root = '/import/vision-ephemeral/fl302/data/VQA'
#         toolkit_root = '/homes/fl302/projects/coco-caption'
#     else:
#         data_root = '/data/fl302/data/VQA'
#         toolkit_root = '/data/fl302/data/im2txt/coco-caption'
#     return data_root, toolkit_root


class VQAQuestion(object):
    def __init__(self, question_file):
        anns = load_questions(question_file)
        self.image_ids = [info['question_id'] for info in anns]
        # self.real_image_ids = [info['image_id'] for info in anns]
        self.imgToAnns = {}
        for i, info in enumerate(anns):
            quest_id = info['question_id']
            quest = info['question']
            self.imgToAnns[quest_id] = [{u'caption': quest, u'id': i, u'image_id': quest_id}]

    def getImgIds(self):
        return self.image_ids

    def subsample(self, K):
        prev_image_ids = np.array(self.image_ids)
        prev_img_to_anns = self.imgToAnns
        base_image_ids = np.array([int(image_id / 1000) for image_id in prev_image_ids], dtype=np.int64)
        unique_image_ids = np.unique(base_image_ids)

        sub_img_to_anns = {}
        sub_image_ids = []
        for image_id in unique_image_ids:
            cands = prev_image_ids[base_image_ids == image_id]
            rand_cands = np.random.choice(cands, size=(K,), replace=False)
            for im_idx in rand_cands:
                sub_img_to_anns[im_idx] = prev_img_to_anns[im_idx]
                sub_image_ids.append(im_idx)
        self.image_ids = sub_image_ids
        self.imgToAnns = sub_img_to_anns


class FilterVQAQuestion(object):
    def __init__(self, question_file, valid_quest_ids):
        anns = load_questions(question_file)
        self.image_ids = valid_quest_ids
        aug_id2index = {info['question_id']: i for i, info in enumerate(anns)}
        # self.real_image_ids = [info['image_id'] for info in anns]
        self.imgToAnns = {}
        for i, aug_id in enumerate(valid_quest_ids):
            # pdb.set_trace()
            idx = aug_id2index[aug_id]
            info = anns[idx]
            quest_id = info['question_id']
            quest = info['question']
            self.imgToAnns[quest_id] = [{u'caption': quest, u'id': i, u'image_id': quest_id}]

    def getImgIds(self):
        return self.image_ids


class VQAAnnotation(object):
    def __init__(self, annotation_file, n_reserved_bits=3):
        annotation = load_questions(annotation_file)
        quest_id2question = {}
        for info in annotation:
            quest_id = info['question_id']
            quest = info['question']
            quest_id2question[quest_id] = quest
        self._question_id2question = quest_id2question
        self._n_reserved_bits = n_reserved_bits
        self._step = pow(10, n_reserved_bits)
        self.imgToAnns = {}
        self.image_ids = []
        self.srcImgIds = {}

    def replicate_annotations(self, seed_question_ids):
        self.imgToAnns = {}
        for i, quest_id in enumerate(seed_question_ids):
            src_qusetion_id = int(quest_id / self._step)
            quest = self._question_id2question[src_qusetion_id]
            self.imgToAnns[quest_id] = [{u'caption': quest, u'id': i, u'image_id': quest_id}]
            self.srcImgIds.update({src_qusetion_id: None})
        self.image_ids = seed_question_ids

    def getRealImgIds(self):
        return self.srcImgIds.keys()

    def getImgIds(self):
        return self.image_ids


class QuestionEvaluator(object):
    def __init__(self, question_file):
        self._gt = VQAQuestion(question_file)
        self._evaluator = None
        self._eval_metric = None
        self._scores = []

    def evaluate(self, res_file):
        res = VQAQuestion(res_file)
        self._evaluator = COCOEvalCap(self._gt, res)
        quest_ids = res.getImgIds()
        self.evalute_worker(quest_ids)

    def evalute_worker(self, image_inds):
        self._evaluator.params['image_id'] = image_inds
        # evaluate results
        self._evaluator.evaluate()


def load_questions(question_file):
    d = json.load(open(question_file, 'r'))
    if 'questions' in d:
        return d['questions']
    else:
        return d


def get_weighting_scheme(metrics, scheme='CIDEr'):
    BCMR = {'Bleu_1': 0.5,
            'Bleu_2': 0.5,
            'Bleu_3': 1.0,
            'Bleu_4': 1.0,
            'CIDEr': 1.0,
            'ROUGE_L': 2.0,
            'METEOR': 5.0,
            }
    CIDER = {'Bleu_1': 0.0,
             'Bleu_2': 0.0,
             'Bleu_3': 0.0,
             'Bleu_4': 0.0,
             'CIDEr': 1.0,
             'ROUGE_L': 0.0,
             'METEOR': 0.0,
             }
    BC = {'Bleu_1': 0.0,
          'Bleu_2': 0.0,
          'Bleu_3': 0.0,
          'Bleu_4': 1.0,
          'CIDEr': 1.0,
          'ROUGE_L': 0.0,
          'METEOR': 0.0,
          }
    scheme2w = {'BCMR': BCMR, 'CIDEr': CIDER, 'BC': BC}
    SCHEME = scheme2w[scheme]
    return np.array([SCHEME[m] for m in metrics], dtype=np.float32)[np.newaxis, :]


def find_matched_questions(results, real_quest_ids, cmp_metric='CIDEr'):
    metrics = [key for key in results[0].keys() if key != 'image_id']
    tot_quests = len(results)
    num_metrics = len(metrics)
    scores = np.zeros([tot_quests, num_metrics], dtype=np.float32)
    gen_real_quest_ids = np.zeros(tot_quests, dtype=np.int64)
    aug_quest_ids = np.zeros(tot_quests, dtype=np.int64)
    metric_w = get_weighting_scheme(metrics, cmp_metric)
    for i, res in enumerate(results):
        aug_quest_id = res['image_id']
        aug_quest_ids[i] = aug_quest_id
        real_quest_id = int(aug_quest_id / 1000)
        gen_real_quest_ids[i] = real_quest_id
        for j, m in enumerate(metrics):
            scores[i, j] = res[m]

    valid_scores = []
    filtered_result_inds = []
    for quest_id in real_quest_ids:
        sel_tab = quest_id == gen_real_quest_ids
        if not np.any(sel_tab):
            print('Empty')
            pdb.set_trace()
        cand_scores = scores[sel_tab]
        cand_cmp_score = (cand_scores * metric_w).sum(axis=1)
        max_idx = cand_cmp_score.argmax()
        cand_quest_ids = aug_quest_ids[sel_tab]
        # pdb.set_trace()
        aug_quest_id = cand_quest_ids[max_idx]
        valid_scores.append(cand_scores[max_idx][np.newaxis, :])
        filtered_result_inds.append(aug_quest_id)
    # valid_scores = np.concatenate(valid_scores, axis=0)
    # mean_scores = valid_scores.mean(axis=0)
    # for m, s in zip(metrics, mean_scores):
    #     print('%s: %0.4f' % (m, s))
    return filtered_result_inds


def dump_results(results, res_file):
    import os
    dump_file = os.path.splitext(res_file)[0] + '_oracle_dump.json'
    score_list = []
    for i, res in enumerate(results):
        aug_quest_id_i = res['image_id']
        # try:
        score_i = float(res['CIDEr'])
        # except Exception as e:
        #     print(str(e))
        #     import pdb
        #     pdb.set_trace()
        score_list.append({'aug_quest_id': aug_quest_id_i,
                           'CIDEr': score_i})
    from util import save_json
    save_json(dump_file, score_list)


def evaluate_oracle(res_file, K=None, eval_multiple=False, split='val'):
    def parse_evaluator_scores(_evaluator):
        metrics = ['Bleu_4', 'CIDEr']
        scores = np.array([_evaluator.eval[m] for m in metrics])
        return scores[np.newaxis, :].astype(np.float32)

    res_base = VQAQuestion(res_file)
    res = subsample_questions(res_base, K)
    quest_ids = res.image_ids
    # set ground truth
    vqa_data_root, _ = get_dataset_root()
    anno_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root,
                                                                         split)
    gt = VQAAnnotation(anno_file)
    gt.replicate_annotations(quest_ids)

    # average test
    evaluator = COCOEvalCap(gt, res)
    evaluator.setup_scorer(['Bleu', 'CIDEr'])
    evaluator.evaluate()
    results = evaluator.evalImgs

    # dump results
    dump_results(results, res_file)
    avg_scores = parse_evaluator_scores(evaluator)

    # find oracle
    matched_inds = find_matched_questions(results, gt.getRealImgIds(), 'BC')
    flt_res = FilterVQAQuestion(res_file, matched_inds)

    gt_ = VQAAnnotation(anno_file)
    gt_.replicate_annotations(flt_res.image_ids)

    evaluator = COCOEvalCap(gt_, flt_res)
    evaluator.setup_scorer(['Bleu', 'CIDEr'])
    evaluator.evaluate()
    oracle_scores = parse_evaluator_scores(evaluator)
    if eval_multiple:
        return avg_scores, oracle_scores  # average and mean scores
    else:
        return oracle_scores.flatten()


def subsample_questions(res, K=None):
    if K is None:
        return res
    else:
        from copy import deepcopy
        new_res = deepcopy(res)
        new_res.subsample(K)
        return new_res


def evaluate_ground_truth():
    # set ground truth
    vqa_data_root, _ = get_dataset_root()
    anno_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root,
                                                                         'val')
    gt = VQAQuestion(anno_file)
    res = VQAQuestion(anno_file)

    # initial test
    evaluator = COCOEvalCap(gt, res)
    evaluator.evaluate()

    results = evaluator.evalImgs

    return evaluator.eval['CIDEr']


def show_trade_off_plot(res_file):
    num_trials = 5
    K = [1, 5, 10, 20, 30, 40, 50, 100]
    # K = [1, 2]

    def evalute_one_config(k, prefix='', _num_trials=5):
        scores_avg, scores_oracle = [], []
        for i in range(_num_trials):
            print('%srandomise %d/%d' % (prefix, i + 1, _num_trials))
            avg, oracle = evaluate_oracle(res_file, k, eval_multiple=True)
            scores_avg.append(avg)
            scores_oracle.append(oracle)

        def get_average_scores(scores):
            scores_ = np.concatenate(scores)
            score_mean = scores_.mean(axis=0)
            score_std = scores_.std(axis=0)
            return score_mean, score_std

        mean_avg, std_avg = get_average_scores(scores_avg)
        mean_oracle, std_oracle = get_average_scores(scores_oracle)
        return (mean_avg, std_avg), (mean_oracle, std_oracle)

    stds_avg, stds_oracle = [], []
    means_avg, means_oracle = [], []

    for k in K:
        _n_t = 1 if k == K[-1] else num_trials
        avg_t, oracle_t = evalute_one_config(k, 'Eval k=%d: ' % k, _n_t)
        # average
        means_avg.append(avg_t[0])
        stds_avg.append(avg_t[1])
        # oracle
        means_oracle.append(oracle_t[0])
        stds_oracle.append(oracle_t[1])

    means_avg = np.array(means_avg)
    stds_avg = np.array(stds_avg)
    means_oracle = np.array(means_oracle)
    stds_oracle = np.array(stds_oracle)
    from scipy.io import savemat
    savemat('result/mean_scores_p0.mat', {'k': K,
                                          'mu_avg': means_avg,
                                          'sigma_avg': stds_avg,
                                          'mu_oracle': means_oracle,
                                          'sigma_oracle': stds_oracle})


if __name__ == '__main__':
    # evaluate_ground_truth()
    res_file = '/import/vision-datasets001/fl302/code/inverse_vqa/result/aug_var_vaq_kl0_greedy_VAQ-VAR.json'
    # evaluate_oracle(res_file, 20)
    show_trade_off_plot(res_file)
