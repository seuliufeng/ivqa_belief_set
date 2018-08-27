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
from post_process_variation_questions import process_one, wrap_samples_for_language_model
from models.language_model import LanguageModel
from time import time
# from n2mn_wrapper import N2MNWrapper
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
tf.flags.DEFINE_string("method", "vae_ia_lmonly_full",
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


def ivqa_decoding_beam_search(checkpoint_path=None):
    model_config = ModelConfig()
    method = FLAGS.method
    res_file = 'result/bs_cand_for_vis.json'
    # Get model
    model_fn = get_model_creation_fn('VAQ-Var')
    create_fn = create_reader('VAQ-VVIS', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file='../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt')

    # get data reader
    subset = 'kpval'
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
        model.set_num_sampling_points(5000)
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

        # build VQA model
    # vqa_model = N2MNWrapper()
    # vqa_model = MLBWrapper()
    num_batches = reader.num_batches

    quest_ids_to_vis = {5682052: 'bread',
                        965492: 'plane',
                        681282: 'station'}

    print('Running beam search inference...')
    results = []
    batch_vqa_scores = []

    num = FLAGS.max_iters if FLAGS.max_iters > 0 else num_batches
    for i in range(num):

        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        quest_id_key = int(quest_ids)

        if quest_id_key not in quest_ids_to_vis:
            continue
        # pdb.set_trace()

        im, gt_q, _, top_ans, ans_tokens, ans_len = outputs[:-2]
        # pdb.set_trace()
        if top_ans == 2000:
            continue

        print('\n%d/%d' % (i, num))
        question_id = int(quest_ids[0])
        image_id = int(image_ids[0])

        t1 = time()
        pathes, scores = model.greedy_inference([im, ans_tokens, ans_len], sess)

        # find unique
        ivqa_scores, ivqa_pathes = process_one(scores, pathes)
        t2 = time()
        print('Time for sample generation: %0.2fs' % (t2 - t1))

        # apply language model
        language_model_inputs = wrap_samples_for_language_model([ivqa_pathes],
                                                                pad_token=model.pad_token - 1,
                                                                max_length=20)
        match_gt = exemplar.query(ivqa_pathes)
        legality_scores = language_model.inference(language_model_inputs)
        legality_scores[match_gt] = 1.0
        num_keep = max(100, (legality_scores > 0.1).sum())  # no less than 100
        valid_inds = (-legality_scores).argsort()[:num_keep]
        print('keep: %d/%d' % (num_keep, len(ivqa_pathes)))

        t3 = time()
        print('Time for language model filtration: %0.2fs' % (t3 - t2))

        def token_arr_to_list(arr):
            return arr.flatten().tolist()

        for _pid, idx in enumerate(valid_inds):
            path = ivqa_pathes[idx]
            # sc = vqa_scores[idx]
            sentence = to_sentence.index_to_question(path)
            aug_quest_id = question_id * 1000 + _pid
            res_i = {'image_id': int(image_id),
                     'aug_id': aug_quest_id,
                     'question_id': question_id,
                     'target': sentence,
                     'top_ans_id': int(top_ans),
                     'question': to_sentence.index_to_question(token_arr_to_list(gt_q)),
                     'answer': to_sentence.index_to_answer(token_arr_to_list(ans_tokens))}
            results.append(res_i)

    save_json(res_file, results)
    return None


def main(_):
    ivqa_decoding_beam_search(None)


if __name__ == '__main__':
    tf.app.run()
