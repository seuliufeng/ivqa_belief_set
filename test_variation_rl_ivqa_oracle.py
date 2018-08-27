from __future__ import division
import tensorflow as tf

import numpy as np
import pylab as plt
import os
from skimage.io import imread
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from eval_vqa_question_oracle import evaluate_oracle
from post_process_variation_questions import post_process_variation_questions

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id
IM_ROOT = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco/'

tf.flags.DEFINE_string("model_type", "VAQ-VarRL",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_var_ivqa_restval_%s",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def ivqa_decoding_beam_search(checkpoint_path=None, subset='kpval'):
    model_config = ModelConfig()
    res_file = 'result/aug_var_vaq_kl0_greedy_%s.json' % FLAGS.model_type.upper()
    # Get model
    model_fn = get_model_creation_fn('VAQ-Var')
    create_fn = create_reader('VAQ-Var', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=50, subset=subset,
                       version=FLAGS.test_version)

    if checkpoint_path is None:
        ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
        # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL/'
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        checkpoint_path = ckpt.model_checkpoint_path

    # Build model
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'sampling')
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')
    results = []
    for i in range(num_batches):
        print('iter: %d/%d' % (i, num_batches))
        if i >= 100:
            break
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        scores, pathes = model.greedy_inference(outputs[:-2], sess)

        # wrap inputs
        _this_batch_size = quest_ids.size
        dummy_scores = np.zeros_like(pathes, dtype=np.float32)
        _, ivqa_pathes = post_process_variation_questions(dummy_scores, pathes, _this_batch_size)
        # scores, pathes = convert_to_unique_questions(scores, pathes)

        for _q_idx, ps in enumerate(ivqa_pathes):
            image_id = image_ids[_q_idx]
            question_id = int(quest_ids[_q_idx])

            for _p_idx, p in enumerate(ps):
                sentence = to_sentence.index_to_question(p)
                aug_quest_id = question_id * 1000 + _p_idx
                res_i = {'image_id': int(image_id),
                         'question_id': aug_quest_id,
                         'question': sentence}
                results.append(res_i)

    save_json(res_file, results)
    return res_file


def main(_):
    from watch_model import ModelWatcher
    subset = 'kpval'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = ivqa_decoding_beam_search(checkpoint_path=model_path,
                                                 subset=subset)
            cider = evaluate_oracle(res_file)
        return float(cider)

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL_ft/'
    # res_file = ivqa_decoding_beam_search(None,
    #                                      subset=subset)
    # print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    tf.app.run()
