from __future__ import division
import tensorflow as tf
import numpy as np
import os
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from util import save_json
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from post_process_variation_questions import process_one, wrap_samples_for_language_model, put_to_array
from models.language_model import LanguageModel
from time import time
import pdb

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id
IM_ROOT = '/usr/data/fl302/data/VQA/Images/'

tf.flags.DEFINE_string("model_type", "VAQ-Var",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_pat", "model/%s_var_ivqa_restval_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory to model checkpoints.")
tf.flags.DEFINE_string("method", "vae_ia",
                       "Name of current model.")
tf.flags.DEFINE_integer("max_iters", 0,
                        "How many samples to evaluate, set 0 to use all of them")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


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

    def query(self, samples):
        is_gt = []
        for p in samples:
            _key = self.serialize_path(p)
            is_gt.append(_key in self.gt_keys)
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


def serialize_path(path):
    return ' '.join([str(t) for t in path])


def ivqa_decoding_beam_search(checkpoint_path=None):
    model_config = ModelConfig()
    method = FLAGS.method
    res_file = 'result/bs_gen_%s.json' % method
    score_file = 'result/bs_vqa_scores_%s.mat' % method
    # Get model
    model_fn = get_model_creation_fn('VAQ-Var')
    create_fn = create_reader('VAQ-VVIS', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    subset = 'kptrain'
    reader = create_fn(batch_size=1, subset=subset,
                       version=FLAGS.test_version)

    exemplar = ExemplarLanguageModel()

    if checkpoint_path is None:
        if FLAGS.checkpoint_dir:
            ckpt_dir = FLAGS.checkpoint_dir
        else:
            ckpt_dir = FLAGS.checkpoint_pat % (FLAGS.version, FLAGS.model_type)
        # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL/'
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        checkpoint_path = ckpt.model_checkpoint_path

    # Build model
    g = tf.Graph()
    with g.as_default():
        # Build the model.ex
        model = model_fn(model_config, 'sampling')
        model.set_num_sampling_points(5)
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

        # build language model
        language_model = LanguageModel()
        language_model.build()
        language_model.set_cache_dir('test_empty')
        # language_model.set_cache_dir('v1_var_att_lowthresh_cache_restval_VAQ-VarRL')
        language_model.set_session(sess)
        language_model.setup_model()

    num_batches = reader.num_batches

    print('Running beam search inference...')

    num = FLAGS.max_iters if FLAGS.max_iters > 0 else num_batches
    neg_pathes = []
    need_stop = False
    for i in range(num):

        outputs = reader.get_test_batch()

        # inference
        im, _, _, top_ans, ans_tokens, ans_len = outputs[:-2]
        if top_ans == 2000:
            continue

        print('\n%d/%d' % (i, num))

        t1 = time()
        pathes, scores = model.greedy_inference([im, ans_tokens, ans_len], sess)

        # find unique
        ivqa_scores, ivqa_pathes = process_one(scores, pathes)
        t2 = time()
        print('Time for sample generation: %0.2fs' % (t2 - t1))

        # apply language model
        language_model_inputs = wrap_samples_for_language_model([ivqa_pathes],
                                                                pad_token=model.pad_token-1,
                                                                max_length=20)
        match_gt = exemplar.query(ivqa_pathes)
        legality_scores = language_model.inference(language_model_inputs)
        legality_scores[match_gt] = 1.0

        neg_inds = np.where(legality_scores < 0.2)[0]
        for idx in neg_inds:
            ser_neg = serialize_path(ivqa_pathes[idx][1:])
            neg_pathes.append(ser_neg)
            if len(neg_pathes) > 100000:
                need_stop = True
                break
            # if len(neg_pathes) > 1000:
            #     need_stop = True
            #     break
            # print('Neg size: %d' % len(neg_pathes))
        if need_stop:
            break
    sv_file = 'data/lm_init_neg_pathes.json'
    save_json(sv_file, neg_pathes)


def main(_):
    # FLAGS.ckpt_file
    ivqa_decoding_beam_search(None)


if __name__ == '__main__':
    tf.app.run()
