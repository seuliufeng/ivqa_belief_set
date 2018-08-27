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
tf.flags.DEFINE_string("method", "vae_ia_vanflt_full",
                       "Name of current model.")
tf.flags.DEFINE_integer("max_iters", 0,
                        "How many samples to evaluate, set 0 to use all of them")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


class VQAWrapper(object):
    def __init__(self, g, sess):
        from models.vqa_base import BaseModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.sess = sess
        ckpt_file = 'model/kprestval_VQA-BaseNorm/model.ckpt-26000'
        with g.as_default():
            self.sess = tf.Session()

            self.model = BaseModel(config, phase='test')
            with tf.variable_scope('VQA'):
                self.model.build()
            # vars = tf.trainable_variables()
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VQA')
            vars_dict = {v.name.replace('VQA/', '').split(':')[0]: v for v in vars}
            self.saver = tf.train.Saver(var_list=vars_dict)
            self.saver.restore(self.sess, ckpt_file)

    def get_scores(self, sampled, image, top_ans_id):
        pathes = []
        for p in sampled:
            if p[-1] == END_TOKEN:
                pathes.append(p[1:-1])  # remove start end token
            else:
                pathes.append(p[1:])  # remove start end token
        num_unk = len(sampled)

        images_aug = np.tile(image, [num_unk, 1])

        # put to arrays
        arr, arr_len = put_to_array(pathes)
        scores = self.model.inference(self.sess, [images_aug, arr, arr_len])
        pred_labels = scores.argmax(axis=1)
        vqa_scores = scores[:, top_ans_id].flatten()
        return vqa_scores, pred_labels == top_ans_id


class MLBWrapper(object):
    def __init__(self, ckpt_file='model/v1_vqa_VQA/v1_vqa_VQA_best2/model.ckpt-135000'):
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        from models.vqa_soft_attention import AttentionModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.name = ' ------- MLB-attention ------- '

        with self.g.as_default():
            self.sess = tf.Session()
            self.model = AttentionModel(config, phase='test_broadcast')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

    def _load_image(self, image_id):
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        filename = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]

    def get_scores(self, sampled, image_id, top_ans_id):
        # process questions
        pathes = []
        for p in sampled:
            if p[-1] == END_TOKEN:
                pathes.append(p[1:-1])  # remove start end token
            else:
                pathes.append(p[1:])  # remove start end token
        num_unk = len(sampled)
        arr, arr_len = put_to_array(pathes)

        # load image
        res5c = self._load_image(image_id)
        images_aug = np.tile(res5c, [num_unk, 1, 1, 1])

        # inference
        scores = self.model.inference(self.sess, [images_aug, arr, arr_len])
        vqa_scores = scores[:, top_ans_id].flatten()
        return vqa_scores


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
    res_file = 'result/bs_gen_%s.json' % method
    score_file = 'result/bs_vqa_scores_%s.mat' % method
    # Get model
    model_fn = get_model_creation_fn('VAQ-Var')
    create_fn = create_reader('VAQ-VVIS', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    subset = 'kptest'
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
        model.set_num_sampling_points(1000)
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
        vqa_model = VQAWrapper(g, sess)
    # vqa_model = MLBWrapper()
    num_batches = reader.num_batches

    print('Running beam search inference...')
    results = []
    batch_vqa_scores = []

    num = FLAGS.max_iters if FLAGS.max_iters > 0 else num_batches
    for i in range(num):

        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        im, _, _, top_ans, ans_tokens, ans_len = outputs[:-2]
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

        t3 = time()
        print('Time for language model filtration: %0.2fs' % (t3 - t2))

        # for idx in valid_inds:
        #     path = ivqa_pathes[idx]
        #     sc = legality_scores[idx]
        #     sentence = to_sentence.index_to_question(path)
        #     # questions.append(sentence)
        #     print('%s (%0.3f)' % (sentence, sc))

        # apply  VQA model
        sampled = [ivqa_pathes[_idx] for _idx in valid_inds]
        # vqa_scores = vqa_model.get_scores(sampled, image_id, top_ans)
        vqa_scores, is_valid = vqa_model.get_scores(sampled, im, top_ans)
        # conf_inds = (-vqa_scores).argsort()[:20]
        conf_inds = np.where(is_valid)[0]
        # pdb.set_trace()
        # conf_inds = (-vqa_scores).argsort()[:40]

        t4 = time()
        print('Time for VQA verification: %0.2fs' % (t4 - t3))

        this_mean_vqa_score = vqa_scores[conf_inds].mean()
        print('sampled: %d, unique: %d, legal: %d, gt: %d, mean score %0.2f' %
              (pathes.shape[0], len(ivqa_pathes), num_keep, match_gt.sum(),
               this_mean_vqa_score))
        batch_vqa_scores.append(this_mean_vqa_score)

        for _pid, idx in enumerate(conf_inds):
            path = sampled[idx]
            sc = vqa_scores[idx]
            sentence = to_sentence.index_to_question(path)
            aug_quest_id = question_id * 1000 + _pid
            res_i = {'image_id': int(image_id),
                     'question_id': aug_quest_id,
                     'question': sentence,
                     'score': float(sc)}
            results.append(res_i)

    save_json(res_file, results)
    batch_vqa_scores = np.array(batch_vqa_scores, dtype=np.float32)
    mean_vqa_score = batch_vqa_scores.mean()
    from scipy.io import savemat
    savemat(score_file, {'scores': batch_vqa_scores, 'mean_score': mean_vqa_score})
    print('BS mean VQA score: %0.3f' % mean_vqa_score)
    return res_file, mean_vqa_score


def main(_):
    # FLAGS.ckpt_file
    res_file, mean_vqa_score = ivqa_decoding_beam_search(None)
    from eval_vqa_question_oracle import evaluate_oracle
    evaluate_oracle(res_file)
    print('BS mean VQA score: %0.3f' % mean_vqa_score)


if __name__ == '__main__':
    tf.app.run()
