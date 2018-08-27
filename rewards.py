import tensorflow as tf
import numpy as np
from models.vqa_soft_attention import AttentionModel as VQAAgent
from vqa_config import ModelConfig
from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
from config import VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from bleu_eval.bleu import Bleu
from collections import OrderedDict

# import sys
# sys.path.append("cider")
# from pyciderevalcap.ciderD.ciderD import CiderD

import pdb

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id

_SENT = SentenceGenerator(trainset='trainval')


def _parse_pred_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[1:clen - 1])
        # seqs.append(c[1:clen])
    return seqs


# def _parse_gt_questions(capt, capt_len):
#     seqs = []
#     batch_size = capt.shape[0]
#     pad = np.ones([batch_size, 1], dtype=np.int32) * END_TOKEN
#     capt = np.concatenate([capt, pad], axis=1)
#     for c, clen in zip(capt, capt_len):
#         tmp = c[:clen+1]
#         tmp[-1] = END_TOKEN
#         seqs.append(tmp)
#     return seqs


def _parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen])
    return seqs


def warp_predictions(quest_ids, sampled):
    pred_seqs = _parse_pred_questions(*sampled)
    # print(_SENT.index_to_question(pred_seqs[0]))
    sam_tokens = [' '.join([str(t) for t in path]) for path in pred_seqs]
    return [{'image_id': str(quest_id), 'caption': [capt]} for quest_id, capt in zip(quest_ids, sam_tokens)]


def warp_ground_truth(quest_ids, gts):
    gt_seqs = _parse_gt_questions(*gts)
    gt_tokens = [' '.join([str(t) for t in path]) for path in gt_seqs]
    return OrderedDict([(str(quest_id), [capt]) for quest_id, capt in zip(quest_ids, gt_tokens)])


def warp_prediction_for_bleu(quest_ids, gts):
    gt_seqs = _parse_pred_questions(*gts)
    gt_tokens = [' '.join([str(t) for t in path]) for path in gt_seqs]
    return OrderedDict([(str(quest_id), [capt]) for quest_id, capt in zip(quest_ids, gt_tokens)])


def compute_rewards(scorer, sampled, gts):
    dummy_ids = np.arange(sampled[1].size, dtype=np.int32)
    w_gts = warp_ground_truth(dummy_ids, gts)

    if scorer.method() == 'Cider':
        w_res = warp_predictions(dummy_ids, sampled)
        _, scores = scorer.evaluate(w_gts, w_res)
    else:
        w_res = warp_prediction_for_bleu(dummy_ids, sampled)
        _, scores = scorer.evaluate(w_gts, w_res)
        # scores = np.array(scores)[-1].astype(np.float32)
        scores = np.array(scores).mean(axis=0).astype(np.float32)
    # _, scores = scorer.compute_score(w_gts, w_res)
    return scores


def post_process_prediction(pathes, baseline):
    is_end_token = np.equal(pathes, END_TOKEN)
    max_allowed_length = pathes.shape[1] - 1
    pred_len = np.argmax(is_end_token, axis=1)
    exceed_max_len = np.logical_not(np.any(is_end_token, axis=1))
    pred_len[exceed_max_len] = pathes.shape[1]
    # clip by max length
    pred_len = np.minimum(max_allowed_length, np.argmax(is_end_token, axis=1))
    # slice to max len
    max_len = pred_len.max()
    pathes = pathes[:, :max_len]
    baseline = baseline[:, :max_len + 1]  # need to compute reward of end token
    return pathes, pred_len, baseline


class IVQARewards(object):
    def __init__(self, graph=None, sess=None, use_vqa_reward=False,
                 metric='cider'):
        self.graph = graph
        self.sess = sess
        self.gamma = 0.0
        self.use_vqa_reward = use_vqa_reward and self.gamma > 0
        # self.cider_scorer = ciderEval('ivqa_train_idxs')
        if metric == 'cider':
            self.scorer = ciderEval('v2_ivqa_train_idxs')
        elif metric == 'bleu':
            self.scorer = Bleu(n=4)
        # self.cider_scorer = CiderD(df='v2_ivqa_train_idxs')
        if self.use_vqa_reward:
            with graph.as_default():
                self._build_vqa_agent()

    def _build_vqa_agent(self):
        with tf.variable_scope('vqa_agent'):
            self.vqa_agent = VQAAgent(config=ModelConfig(),
                                      phase='test')

    def get_reward(self, sampled, gts, answers=None):
        """
        compute rewards given a sampled sentence and gt, the reward is
        computed based on CIDEr-D
        :param sampled: a list of sampled samples, [seq, seq_len]
        :param gts: a list of ground-truth samples [seq, seq_len]
        :param answers: numpy.array of ground-truth top answer index
        of VQA
        :return: numpy array of size (N,) of reward for each sample
        """
        rewards = compute_rewards(self.scorer, sampled, gts)
        if self.use_vqa_reward:
            vqa_rewards = self._compute_vqa_reward(sampled, answers)
            rewards = (1. - self.gamma) * rewards + self.gamma * vqa_rewards
        return rewards

    def _compute_vqa_reward(self, sampled, answers):
        probs = self.vqa_agent.prob
        feed_dict = self.vqa_agent.fill_feed_dict(sampled +
                                                  [answers])
        preds = self.sess.run(probs, feed_dict=feed_dict)
        rewards = np.equal(preds.argmax(axis=1), answers).astype(np.float32)
        return rewards
