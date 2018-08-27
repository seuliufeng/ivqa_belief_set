import json
import sys
import numpy as np
from util import get_dataset_root

# def get_dataset_root():
#     import os
#     if os.path.exists('/usr/data/fl302/data/VQA'):
#         data_root = '/usr/data/fl302/data/VQA'
#         toolkit_root = '/usr/data/fl302/toolbox/coco-caption-master/'
#     else:
#         data_root = '/data/fl302/data/VQA'
#         toolkit_root = '/data/fl302/data/im2txt/coco-caption'
#     return data_root, toolkit_root


vqa_data_root, toolkit_dir = get_dataset_root()
sys.path.insert(0, toolkit_dir)
from pycocoevalcap.eval import COCOEvalCap



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


class AnswerTypeManager(object):
    def __init__(self, annotation_file):
        d = load_questions(annotation_file)
        annotation = d['annotations']
        self._answer_type_cands = {}
        self._question_type_cands = {}
        question_ids = []
        for info in annotation:
            question_id = info['question_id']
            question_ids.append(question_id)
            # fill answer filter
            ans_type = info['answer_type']
            if ans_type in self._answer_type_cands:
                self._answer_type_cands[ans_type].append(question_id)
            else:
                self._answer_type_cands[ans_type] = [question_id]
            # fill question filter
            quest_type = info['question_type']
            if quest_type in self._question_type_cands:
                self._question_type_cands[quest_type].append(question_id)
            else:
                self._question_type_cands[quest_type] = [question_id]
        self._answer_type_cands['all'] = question_ids
        self._question_type_cands['all'] = question_ids
        self.answer_types = self._answer_type_cands.keys()
        self.question_types = self._question_type_cands.keys()

    def filter_by_answer_type(self, type, image_ids):
        candidates = np.array(image_ids)
        cand_pool = np.array(self._answer_type_cands[type])
        return np.intersect1d(cand_pool, candidates).tolist()

    def filter_by_question_type(self, type, image_ids):
        candidates = np.array(image_ids)
        cand_pool = np.array(self._question_type_cands[type])
        return np.intersect1d(cand_pool, candidates).tolist()


class QuestionEvaluator(object):
    def __init__(self, annotation_file, question_file):
        self._filter = AnswerTypeManager(annotation_file)
        self._gt = VQAQuestion(question_file)
        self._types = self._filter.answer_types
        self._evaluator = None
        self._eval_metric = None
        self._scores = []

    def evaluate(self, res_file):
        res = VQAQuestion(res_file)
        self._evaluator = COCOEvalCap(self._gt, res)
        quest_ids = res.getImgIds()
        self._types = ['all']
        for ans_type in self._types:
            print('\n====== Evaluate type %s =======' % ans_type.upper())
            self.evalute_subtype(ans_type, quest_ids)
        self._scores = np.array(self._scores).transpose()
        self.print_results()
        return self._scores, self._types, self._eval_metric

    def evalute_subtype(self, type, image_inds):
        inds = self._filter.filter_by_answer_type(type, image_inds)
        self._evaluator.params['image_id'] = inds
        # evaluate results
        self._evaluator.evaluate()
        if self._eval_metric is None:
            self._eval_metric = self._evaluator.eval.keys()
        scores = []
        for metric in self._eval_metric:
            score = self._evaluator.eval[metric]
            scores.append(score)
            print '%s: %.3f' % (metric, score)
        self._scores.append(scores)

    def print_results(self):
        types = '    '.join(self._types)
        print('\t\t%s' % types)
        for metric, score in zip(self._eval_metric, self._scores):
            print('%s\t%s' % (metric, np.array_str(score, precision=3)))

    def save_results(self, eval_res_file='result/alg_eval_result.mat'):
        from scipy.io import savemat
        metric = np.array(self._eval_metric, dtype=np.object)
        types = np.array(self._types, dtype=np.object)
        savemat(eval_res_file, {'metric': metric,
                                'question_type': types,
                                'score': self._scores})

    def get_overall_cider(self):
        return float(self._scores[0][0])

    def get_overall_blue4(self):
        return float(self._scores[2][0])


def load_questions(question_file):
    d = json.load(open(question_file, 'r'))
    if 'questions' in d:
        return d['questions']
    else:
        return d


if __name__ == '__main__':
    subset = 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)
    result_file = '/import/vision-ephemeral/fl302/code/VQA-tensorflow/result/gen_question.json'

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
