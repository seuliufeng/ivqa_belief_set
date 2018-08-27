import numpy as np
from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
from config import VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from bleu_eval.bleu import Bleu
from collections import OrderedDict
import tensorflow as tf
from post_process_variation_questions import put_to_array
from answer_token_to_top_answers import AnswerTokenToTopAnswer
from graph_util import find_connected_components
from uniqueness_reward import UniqueReward
from visual_fact_reward import VisualFactReward
import pdb

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id

_SENT = SentenceGenerator(trainset='trainval')


def _parse_pred_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[1: clen - 1])  # remove start token
    return seqs


def _parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen])
    return seqs


def serialize_path(path):
    return ' '.join([str(t) for t in path])


def supress_cider_score(rewards):
    sup_tab = rewards > 3.
    vals = rewards[sup_tab]
    vals = (vals - 3.0) / 7. + 3.
    rewards[sup_tab] = vals
    return rewards


class TopAnswerVersionConverter(object):
    def __init__(self):
        self.debug = False
        v1_vocab_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        v2_vocab_file = '/usr/data/fl302/code/inverse_vqa/model/mlb_attention_v2/vqa_trainval_top2000_answers.txt'
        v1_vocab = self._load_top_answers(v1_vocab_file)
        v2_vocab = self._load_top_answers(v2_vocab_file)
        v2_vocab2_idx = {word: i for (i, word) in enumerate(v2_vocab)}
        mapping = []
        for word in v1_vocab:
            if word in v2_vocab2_idx:
                mapping.append(v2_vocab2_idx[word])
            else:
                mapping.append(2000)
        mapping.append(2000)  # add oov
        self.top_ans_v1_to_v2 = np.array(mapping, dtype=np.int32)
        self.v1_vocab = v1_vocab + ['UNK']
        self.v2_vocab = v2_vocab + ['UNK']

    @staticmethod
    def _load_top_answers(vocab_file):
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
            return [line.strip() for line in reverse_vocab]

    def convert(self, v1_top_ans_ids):
        v2_top_ans_ids = self.top_ans_v1_to_v2[v1_top_ans_ids]
        if self.debug:
            num = v1_top_ans_ids.size
            vis_ids = np.random.choice(num, 3, replace=False)
            for idx in vis_ids:
                print('v1: %s, v2: %s' % (self.v1_vocab[v1_top_ans_ids[idx]],
                      self.v2_vocab[v2_top_ans_ids[idx]]))
        return v2_top_ans_ids


class AttentionVQARewards(object):
    def __init__(self, ckpt_file='/usr/data/fl302/code/inverse_vqa/model/mlb_attention_v2/model.ckpt-170000',
                 use_dis_reward=False):
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        self.v1tov2 = TopAnswerVersionConverter()
        from models.vqa_soft_attention_v2 import AttentionModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.ans2id = AnswerTokenToTopAnswer()
        self.use_dis_reward = use_dis_reward
        with self.g.as_default():
            self.sess = tf.Session()
            self.model = AttentionModel(config, phase='test_broadcast')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

    def process_answers(self, ans, ans_len):
        ans_pathes = _parse_gt_questions(ans, ans_len)
        return self.ans2id.get_top_answer(ans_pathes)

    def get_reward(self, sampled, inputs):
        if len(inputs) == 4:
            images, res5c, ans, ans_len = inputs
            top_ans_ids = self.process_answers(ans, ans_len)
        else:
            assert (len(inputs) == 5)
            images, res5c, ans, ans_len, top_ans_ids = inputs
        # version conversion
        top_ans_ids = self.v1tov2.convert(top_ans_ids)
        images_aug = []
        top_ans_ids_aug = []
        answer_aug = []
        answer_len_aug = []
        pathes = []
        for _idx, ps in enumerate(sampled):
            for p in ps:
                if p[-1] == END_TOKEN:
                    pathes.append(p[1:-1])  # remove start end token
                else:
                    pathes.append(p[1:])  # remove start end token
                images_aug.append(images[_idx][np.newaxis, :])
                answer_aug.append(ans[_idx][np.newaxis, :])
                answer_len_aug.append(ans_len[_idx])
                top_ans_ids_aug.append(top_ans_ids[_idx])
        # put to arrays
        arr, arr_len = put_to_array(pathes)
        images_aug = np.concatenate(images_aug)
        answer_aug = np.concatenate(answer_aug).astype(np.int32)
        top_ans_ids_aug = np.array(top_ans_ids_aug)
        answer_len_aug = np.array(answer_len_aug, dtype=np.int32)
        # run inference in VQA
        scores = self.model.inference(self.sess, [res5c, arr, arr_len])
        if self.use_dis_reward:
            vqa_scores = np.require(scores.argmax(axis=1) == top_ans_ids_aug,
                                    np.float32)
        else:
            _this_batch_size = scores.shape[0]
            vqa_scores = scores[np.arange(_this_batch_size), top_ans_ids_aug]
        is_valid = top_ans_ids_aug != 2000
        return vqa_scores, [images_aug, answer_aug, answer_len_aug, is_valid]


class VQARewards(object):
    def __init__(self, ckpt_file='', use_dis_reward=False,
                 use_attention_model=False):
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        self.use_attention_model = use_attention_model
        from models.vqa_base import BaseModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.ans2id = AnswerTokenToTopAnswer()
        self.use_dis_reward = use_dis_reward
        with self.g.as_default():
            self.sess = tf.Session()
            if self.use_attention_model:
                self.model = AttentionModel(config, phase='test')
                self.model.build()
            else:
                self.model = BaseModel(config, phase='test')
                self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

    def process_answers(self, ans, ans_len):
        ans_pathes = _parse_gt_questions(ans, ans_len)
        return self.ans2id.get_top_answer(ans_pathes)

    def get_reward(self, sampled, inputs):
        if len(inputs) == 3:
            images, ans, ans_len = inputs
            top_ans_ids = self.process_answers(ans, ans_len)
        else:
            assert (len(inputs) == 4)
            images, ans, ans_len, top_ans_ids = inputs
        images_aug = []
        top_ans_ids_aug = []
        answer_aug = []
        answer_len_aug = []
        pathes = []
        for _idx, ps in enumerate(sampled):
            for p in ps:
                if p[-1] == END_TOKEN:
                    pathes.append(p[1:-1])  # remove start end token
                else:
                    pathes.append(p[1:])  # remove start end token
                images_aug.append(images[_idx][np.newaxis, :])
                answer_aug.append(ans[_idx][np.newaxis, :])
                answer_len_aug.append(ans_len[_idx])
                top_ans_ids_aug.append(top_ans_ids[_idx])
        # put to arrays
        arr, arr_len = put_to_array(pathes)
        images_aug = np.concatenate(images_aug)
        answer_aug = np.concatenate(answer_aug).astype(np.int32)
        top_ans_ids_aug = np.array(top_ans_ids_aug)
        answer_len_aug = np.array(answer_len_aug, dtype=np.int32)
        # run inference in VQA
        scores = self.model.inference(self.sess, [images_aug, arr, arr_len])
        if self.use_dis_reward:
            vqa_scores = np.require(scores.argmax(axis=1) == top_ans_ids_aug,
                                    np.float32)
        else:
            _this_batch_size = scores.shape[0]
            vqa_scores = scores[np.arange(_this_batch_size), top_ans_ids_aug]
        is_valid = top_ans_ids_aug != 2000
        return vqa_scores, [images_aug, answer_aug, answer_len_aug, is_valid]


class DiversityReward(object):
    def __init__(self, mode='winner_take_all'):
        self.scorer = ciderEval('vqa_%s_idxs_end' % 'kptrain')
        self.pred_has_start_end_token = True
        self.use_end_token = True
        self.thresh = 9.0
        self.verbose = False
        self.mode = mode
        self.diversity_scorer = UniqueReward()
        assert (self.mode in ['winner_take_all', 'kill_all'])

    def set_mode(self, mode):
        self.mode = mode
        assert (self.mode in ['winner_take_all', 'kill_all'])

    def get_reward(self, sampled, scores):
        """
        Winner takes all diversity reward, only winner will be given a reward 1, otherwise 0.
        :param sampled: sampled questions
        :param scores: log likelihood of the sentence
        :return:
        """
        sampled = self.process_sampled(sampled)
        wrapped_ref, wrapped_res, path_ids = self.wrap_sampled_pairwise(sampled)
        _, sim = self.scorer.evaluate(wrapped_ref, wrapped_res)  # cider similarity
        diversity, is_gt = self.diversity_scorer.get_reward(sampled)
        d_rewards = []
        for _ids, _scs, _ps in zip(path_ids, scores, sampled):
            _scs = np.array(_scs)
            num_cand = len(_scs)
            _sim = sim[_ids]

            _d_reward = np.ones(shape=(num_cand,), dtype=np.float32)
            _rows, _cols = np.tril_indices(num_cand, k=-1)

            connect_tab = _sim > self.thresh  # too close
            _edges = [(r, c) for r, c in zip(_rows[connect_tab],
                                             _cols[connect_tab])]
            if _edges:
                _ccs = find_connected_components(_edges)
                for _cc in _ccs:
                    _cc_scores = _scs[_cc]
                    _max_idx = _cc[_cc_scores.argmax()]
                    _d_reward[_cc] = 0.
                    if self.mode == 'winner_take_all':
                        _d_reward[_max_idx] = 1.
            d_rewards.append(_d_reward)
            if _edges and self.verbose:
                self.print_questions(_ps, _d_reward, _scs)
        d_rewards = np.concatenate(d_rewards)
        d_rewards *= diversity
        return d_rewards, is_gt

    def print_questions(self, sampled, rewards, scores):
        for sm, r, sc in zip(sampled, rewards, scores):
            sent = _SENT.index_to_question(sm[:-1])
            print('%s (%0.3f, %0.3f)' % (sent, r, sc))
        print('\n')

    def process_sampled(self, sampled):
        new_sampled = []
        for ps in sampled:
            tmp = []
            for p in ps:
                if self.pred_has_start_end_token:
                    _p = p[1:]
                else:
                    _p = p + [END_TOKEN]
                if not self.use_end_token:
                    _p = _p[:-1]
                tmp.append(_p)
            new_sampled.append(tmp)
        return new_sampled

    @staticmethod
    def wrap_sampled_pairwise(sampled):
        wrapped_ref = OrderedDict()
        wrapped_res = []
        idx = 0
        path_ids = []
        for _var_s in sampled:
            _u_tmp = []
            _rows, _cols = np.tril_indices(len(_var_s), k=-1)

            for _res_id, ref_id in zip(_rows, _cols):
                _key = str(idx)
                wrapped_ref[_key] = [serialize_path(_var_s[ref_id])]
                wrapped_res.append({'image_id': _key,
                                    'caption': [serialize_path(_var_s[_res_id])]})
                _u_tmp.append(idx)
                idx += 1
            path_ids.append(_u_tmp)
        return wrapped_ref, wrapped_res, path_ids


class IVQARewards(object):
    def __init__(self, metric='cider', gt_has_start_end_token=False,
                 pred_has_start_end_token=True, use_end_token=True,
                 subset='kptrain'):
        self.gt_has_start_end_token = gt_has_start_end_token
        self.pred_has_start_end_token = pred_has_start_end_token
        self.use_end_token = use_end_token
        if metric == 'cider':
            self.scorer = ciderEval('vqa_%s_idxs_end' % subset)
        elif metric == 'bleu':
            self.scorer = Bleu(n=4)
        assert (metric == 'cider')
        self.to_sentence = SentenceGenerator(trainset='trainval')
        self._num_call = long(0)
        self.print_iterval = 100

    def get_reward(self, sampled, gts):
        """
        compute rewards given a sampled sentence and gt, the reward is
        computed based on CIDEr-D
        :param sampled: a list of list of pathes
        :param gts: a list of ground-truth samples [seq, seq_len]
        :param answers: numpy.array of ground-truth top answer index
        of VQA
        :return: numpy array of size (N,) of reward for each sample
        """
        gts = self.process_gt(gts)  # convert to list
        sampled = self.process_sampled(sampled)  # convert to list
        wrapped_gt, wrapped_sample = self.wrap_samples(sampled, gts)
        _, rewards = self.scorer.evaluate(wrapped_gt, wrapped_sample)
        # if not self._num_call % self.print_iterval:
        #     self.print_questions(gts, sampled, rewards)
        # self._num_call += 1
        # rewards = supress_cider_score(rewards)
        return rewards / 10.  # normalise to [0-1]

    def print_questions(self, gts, sampled, rewards):
        n_vis = 2
        num_tot = len(gts)
        vis_ids = np.random.choice(num_tot, size=(n_vis,), replace=False)
        offsets = np.cumsum([len(sms) for sms in sampled]).tolist()
        offsets = [0] + offsets
        for _vis_id in vis_ids:
            _gt = gts[_vis_id]
            sent = self.to_sentence.index_to_question(_gt[:-1])
            print('\nGT: %s' % sent)
            _sms = sampled[_vis_id]
            _offset = offsets[_vis_id]
            for _sid, sm in enumerate(_sms):
                _r = rewards[_offset + _sid]
                sent = self.to_sentence.index_to_question(sm[:-1])
                print('%s (%0.3f)' % (sent, _r))
        print('\n')

    @staticmethod
    def wrap_samples(sampled, gts):
        wrapped_gt = OrderedDict()
        wrapped_sample = []
        idx = 0
        for _var_s, _gt in zip(sampled, gts):
            _gt_pat = serialize_path(_gt)
            for _s in _var_s:
                _key = str(idx)
                _s_pat = serialize_path(_s)
                wrapped_gt[_key] = [_gt_pat]
                wrapped_sample.append({'image_id': _key, 'caption': [_s_pat]})
                idx += 1
        return wrapped_gt, wrapped_sample

    def process_gt(self, gts):
        capt, capt_len = gts
        seqs = []
        for c, clen in zip(capt, capt_len):
            _gt = c[:clen].tolist()
            if self.gt_has_start_end_token:
                _gt = _gt[1:]
            else:
                _gt += [END_TOKEN]
            if not self.use_end_token:
                _gt = _gt[:-1]
            seqs.append(_gt)
        return seqs

    def process_sampled(self, sampled):
        new_sampled = []
        for ps in sampled:
            tmp = []
            for p in ps:
                if self.pred_has_start_end_token:
                    _p = p[1:]
                else:
                    _p = p + [END_TOKEN]
                if not self.use_end_token:
                    _p = _p[:-1]
                tmp.append(_p)
            new_sampled.append(tmp)
        return new_sampled


class MixReward(object):
    def __init__(self, thresh=0.3, cider_w=0.6, dis_vqa_reward=False,
                 attention_vqa=False):
        if attention_vqa:
            self.vqa_reward = AttentionVQARewards(use_dis_reward=dis_vqa_reward)
        else:
            self.vqa_reward = VQARewards('model/kprestval_VQA-BaseNorm/model.ckpt-26000',
                                         use_dis_reward=dis_vqa_reward)
        self.cider_reward = VisualFactReward()
        # self.cider_reward = IVQARewards()
        self.diversity_reward = DiversityReward()
        self.thresh = thresh
        self.cider_w = cider_w
        self.to_sentence = SentenceGenerator(trainset='trainval')
        self._num_call = long(0)
        self.print_iterval = 100
        self.language_thresh = 0.2
        self.cider_thresh = 0.05
        self.use_cider = True
        self.lm = None
        self.replay_buffer = None

    def set_language_model(self, model):
        self.lm = model

    def set_replay_buffer(self, insert_thresh=0.5,
                          sv_dir='vqa_replay_buffer'):
        from vqa_replay_buffer import VQAReplayBuffer
        self.replay_buffer = VQAReplayBuffer(insert_thresh=insert_thresh,
                                             sv_dir=sv_dir)

    def cache_questions(self, quest_ids, questions, rewards):
        vqa_reward, _, language_reward, _, _ = rewards
        mask = self.apply_language_mask(language_reward)  # is grammar correct
        self.replay_buffer.insert(quest_ids, questions, vqa_reward * mask)

    def set_cider_state(self, use_cider):
        self.use_cider = use_cider

    def set_language_thresh(self, t):
        self.language_thresh = t

    def compute_lm_reward(self, _lm_inputs):
        return self.lm.inference(_lm_inputs)

    def apply_cider_mask(self, cider_scores):
        return (cider_scores >= self.cider_thresh).astype(np.float32)

    def apply_language_mask(self, language_scores):
        return (language_scores >= self.language_thresh).astype(np.float32)

    def apply_mask(self, rewards):
        [vqa_reward, cider_reward, language_reward, diversity_reward] = rewards
        mask = self.apply_language_mask(language_reward)
        if self.use_cider:
            mask *= self.apply_cider_mask(cider_reward)
        return vqa_reward * mask * diversity_reward

    def get_reward(self, sampled, gts, context):
        diversity_reward, is_gt = self.diversity_reward.get_reward(sampled, context[2])
        vqa_reward, aug_data = self.vqa_reward.get_reward(sampled, context[0])
        cider_reward = self.cider_reward.get_reward(sampled, context[3])  # question ids
        language_reward = self.compute_lm_reward(context[1])
        language_reward[is_gt] = 1.0  # correct language model prediction
        rewards = [vqa_reward, cider_reward, language_reward, diversity_reward]
        overall_reward = self.apply_mask(rewards)
        rewards.append(overall_reward)
        # cache and print questions
        # if self.replay_buffer:
        #     self.cache_questions(context[3], sampled, rewards)
        self.print_questions(_parse_gt_questions(*gts), sampled, rewards)
        rewards = self.concat_rewards(rewards)
        return overall_reward, rewards, is_gt, aug_data

    def concat_rewards(self, inputs):
        return np.concatenate([_in[:, np.newaxis] for _in in inputs], axis=1)

    def print_questions(self, gts, sampled, rewards):
        self._num_call += 1
        if self._num_call % self.print_iterval:
            return

        num_tot = len(gts)
        n_vis = min(2, num_tot)
        r1, r2, r3, r0, r = rewards
        vis_ids = np.random.choice(num_tot, size=(n_vis,), replace=False)
        offsets = np.cumsum([len(sms) for sms in sampled]).tolist()
        offsets = [0] + offsets
        for _vis_id in vis_ids:
            _gt = gts[_vis_id]
            sent = self.to_sentence.index_to_question(_gt)
            print('\nGT: %s' % sent)
            _sms = sampled[_vis_id]
            _offset = offsets[_vis_id]
            for _sid, sm in enumerate(_sms):
                _r0 = r0[_offset + _sid]
                _r1 = r1[_offset + _sid]
                _r2 = r2[_offset + _sid]
                _r3 = r3[_offset + _sid]
                _r = r[_offset + _sid]
                sent = self.to_sentence.index_to_question(sm[1:-1])
                print('%s (vqa:%0.3f, cider:%0.3f, lm:%0.3f, diver: %0.3f, overall:%0.3f)' %
                      (sent, _r1, _r2, _r3, _r0, _r))
        print('\n')

        # def get_reward_(self, sampled, gts, context):
        #     diversity_reward, is_gt = self.diversity_reward.get_reward(sampled, context[2])
        #     vqa_reward, aug_data = self.vqa_reward.get_reward(sampled, context[0])
        #     cider_reward = self.cider_reward.get_reward(sampled, gts)  # cider
        #     language_reward = self.compute_lm_reward(context[1])
        #     language_reward[is_gt] = 1.0
        #
        #     # rewards = self.diversity_reward.get_reward(sampled)
        #     # lm_mask = np.logical_and(language_reward < 0.4, cider_reward < 3.)
        #     #  not match, not valid, promote valid questions
        #     tmp = vqa_reward.copy()
        #     if self.language_thresh is None:
        #         lm_mask = language_reward
        #     else:
        #         rm_mask = np.logical_and(language_reward < self.language_thresh,
        #                                  cider_reward < 3.)  # not match, not valid, promote valid questions
        #         lm_mask = (1.0 - rm_mask)
        #         # tmp[lm_mask] = 0.
        #     tmp *= lm_mask
        #     if self.use_cider:
        #         r = np.maximum(cider_reward, tmp)
        #     else:
        #         r = tmp
        #     r *= diversity_reward  # discourage duplicate
        #     rewards = [vqa_reward, cider_reward, language_reward, diversity_reward, r]
        #     self.print_questions(_parse_gt_questions(*gts), sampled, rewards)
        #     rewards = self.concat_rewards(rewards)
        #     return r, rewards, is_gt, aug_data
