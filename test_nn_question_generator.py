import numpy as np
from time import time
from util import load_json, load_hdf5, save_json
from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
from inference_utils.question_generator_util import SentenceGenerator


def _parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen])
    return seqs


def wrap_candidates(quest_ids, res):
    # warp gts
    w_gts = {str(quest_id): res for quest_id in quest_ids}
    w_res = [{'image_id': str(quest_id), 'caption': [capt]} for quest_id, capt in zip(quest_ids, res)]
    return w_gts, w_res


def wrap_ranking_candidates(choices, neighs):
    dummy_ids = np.arange(len(choices), dtype=np.int32)
    # warp gts
    w_gts = {str(quest_id): neighs for quest_id in dummy_ids}
    w_res = [{'image_id': str(quest_id), 'caption': [capt]} for quest_id, capt in zip(dummy_ids, choices)]
    return w_gts, w_res


class QuestionPool(object):
    def __init__(self):
        print('Creating NN model')
        subset = 'kptrain'
        meta_file = 'data/vqa_std_mscoco_%s.meta' % subset
        data_file = 'data/vqa_std_mscoco_%s.data' % subset
        d = load_json(meta_file)
        quest_ids = d['quest_id']
        self.quest_id2index = {quest_id: i for (i, quest_id) in enumerate(quest_ids)}
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self.cider_scorer = ciderEval('ivqa_train_idxs')
        print('Done')

    def get_candidates(self, quest_ids):
        index = np.array([self.quest_id2index[quest_id] for quest_id in quest_ids])
        capt = self._quest[index]
        capt_len = self._quest_len[index]
        cands = _parse_gt_questions(capt, capt_len)
        # combine together
        res_token = [' '.join([str(t) for t in path]) for path in cands]
        # wrap
        w_gts, w_res = wrap_candidates(quest_ids, res_token)
        # compute cider
        _, scores = self.cider_scorer.evaluate(w_gts, w_res)
        # print(score)
        # find the one with max cider
        idx = scores.argmax()
        nn_quest_id = quest_ids[idx]
        nn_path = cands[idx]
        return nn_quest_id, nn_path

    def compute_ranking_score(self, cands, pool_ids):
        index = np.array([self.quest_id2index[quest_id] for quest_id in pool_ids])
        capt = self._quest[index]
        capt_len = self._quest_len[index]
        nn_cands = _parse_gt_questions(capt, capt_len)
        # combine together
        nn_token = [' '.join([str(t) for t in path]) for path in nn_cands]
        query_token = [' '.join([str(t) for t in path]) for path in cands]

        w_gts, w_res = wrap_ranking_candidates(query_token, nn_token)
        # compute cider
        _, scores = self.cider_scorer.evaluate(w_gts, w_res)
        return scores


class NNModel(object):
    def __init__(self, k=40, subset='val'):
        val_qids, nn_ids = load_image_nn(subset)
        self.quest_id2index = {qid: i for (i, qid) in enumerate(val_qids)}
        self.nn_ids = nn_ids
        self.k = k
        self.scorer = QuestionPool()

    def score_multi_choices(self, quest_id, capt, capt_len):
        choices = _parse_gt_questions(capt, capt_len)
        index = self.quest_id2index[quest_id]
        nn_qids = self.nn_ids[index, :self.k]
        scores = self.scorer.compute_ranking_score(choices, nn_qids)
        return scores


def load_image_nn(subset='val'):
    from scipy.io import loadmat
    d = loadmat('/import/vision-datasets001/fl302/code/nn_cap/kp%s_im_ans_nn.mat' % subset)
    nn_ids = d['nn_ids'].astype(np.int32)
    val_quest_ids = d['qids_val'].astype(np.int32).flatten()
    return val_quest_ids, nn_ids


def evaluate_question(result_file, subset='kpval', version='v1'):
    from analysis.eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()

    subset = 'train' if subset == 'train' else 'val'
    if version == 'v1':
        annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
        question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)
    elif version == 'v2':
        anno_dir = '/import/vision-ephemeral/fl302/data/VQA2.0'
        annotation_file = '%s/v2_mscoco_%s2014_annotations.json' % (anno_dir, subset)
        question_file = '%s/v2_OpenEnded_mscoco_%s2014_questions.json' % (anno_dir, subset)
    else:
        raise Exception('unknown version, v1 or v2')

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    # return evaluator.get_overall_blue4()
    return evaluator.get_overall_cider()


def main():
    # params
    k = 80
    res_file = 'result/quest_vaq_nn.json'
    # sentence generator
    to_sentence = SentenceGenerator(trainset='trainval')
    # load distances
    val_qids, nn_ids = load_image_nn()
    # create nn model
    nn_model = QuestionPool()
    num = len(val_qids)
    results = []
    for i, (v_qid, v_nn) in enumerate(zip(val_qids, nn_ids)):
        # run nn search
        t = time()
        tr_qid, tr_path = nn_model.get_candidates(v_nn[:k])
        sent = to_sentence.index_to_question(tr_path)
        print(sent)
        print('Processing %d/%d, time %0.2f sec.' % (i, num, time() - t))
        res_i = {'question_id': int(v_qid), 'question': sent}
        results.append(res_i)

    save_json(res_file, results)
    cider = evaluate_question(res_file, subset='kpval',
                              version='v1')


if __name__ == '__main__':
    main()
