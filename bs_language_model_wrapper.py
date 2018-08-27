from __future__ import division
import tensorflow as tf
import numpy as np
from config import VOCAB_CONFIG
from post_process_variation_questions import wrap_samples_for_language_model
from models.language_model import LanguageModel

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id


class ExemplarLanguageModel(object):
    def __init__(self):
        self.history = {}
        self._init_exemplars('kprestval')
        self._init_exemplars('kptrain')
        self.gt_keys = {k: None for k in self.history.keys()}

    def _init_exemplars(self, subset):
        from util import load_hdf5
        print('Initialising statastics with ground truth')
        d = load_hdf5('data/vqa_std_mscoco_%s.data' % subset)
        gts = self.parse_gt_questions(d['quest_arr'], d['quest_len'])
        # update stat
        self._update_samples(gts, generate_key=True)

    def _update_samples(self, samples, generate_key=False):
        for _key in samples:
            if generate_key:
                _key = self.serialize_path(_key)
            if _key in self.history:
                self.history[_key] += 1.
            else:
                self.history[_key] = 1.

    def query(self, samples, thresh=None):
        is_gt = []
        for p in samples:
            _key = self.serialize_path(p)
            if thresh is None:
                is_gt.append(_key in self.gt_keys)
            else:
                _gt = _key in self.gt_keys and self.history[_key] > thresh
                is_gt.append(_gt)
        is_gt = np.array(is_gt, dtype=np.bool)
        return is_gt

    @staticmethod
    def parse_gt_questions(capt, capt_len):
        seqs = []
        for c, clen in zip(capt, capt_len):
            seqs.append([START_TOKEN] + c[:clen].tolist() + [END_TOKEN])
        return seqs

    @staticmethod
    def serialize_path(path):
        return ' '.join([str(t) for t in path])


class NNLanguageModel(object):
    def __init__(self):
        self._build()

    def _build(self):
        g = tf.Graph()
        with g.as_default():
            # build language model
            language_model = LanguageModel()
            language_model.build()
            language_model.set_cache_dir('test_empty')
            sess = tf.Session()
            language_model.set_session(sess)
            language_model.setup_model()
        self.language_model = language_model

    def inference(self, lm_inputs):
        return self.language_model.inference(lm_inputs)


class LanuageModelWrapper(object):
    def __init__(self, min_count=1):
        self.pad_token = 15954
        self.min_gt_count = min_count
        self.nn_lm = NNLanguageModel()
        self.eg_lm = ExemplarLanguageModel()

    def inference(self, ivqa_pathes):
        language_model_inputs = wrap_samples_for_language_model([ivqa_pathes],
                                                                pad_token=self.pad_token - 1,
                                                                max_length=20)
        match_gt = self.eg_lm.query(ivqa_pathes, self.min_gt_count)
        legality_scores = self.nn_lm.inference(language_model_inputs)
        legality_scores[match_gt] = 1.0
        return legality_scores
