from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

# from models.model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from util import load_hdf5

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "VQA-BaseNorm",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kprestval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test():
    # Load data
    def load_data(fpath):
        d = load_hdf5(fpath)
        return d['quest_ids'], d['ans_preds']

    w = 0.8
    quest_ids, preds1 = load_data('data5/kpval_VQA-BaseNorm_scores.data')
    check_quest_ids, preds2 = load_data('data5/kpval_VQA-BaseNorm_scores_flt.data')
    scores = w * preds1 + (1.0 - w) * preds2

    scores[:, -1] = -1.0
    ans_ids = scores.argmax(axis=1)

    # Create the vocabulary.
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)

    result = [{u'answer': to_sentence.index_to_top_answer(aid),
               u'question_id': qid} for aid, qid in zip(ans_ids, quest_ids)]

    # save results
    tf.logging.info('Saving results')
    res_file = FLAGS.result_format % (FLAGS.version, TEST_SET)
    json.dump(result, open(res_file, 'w'))
    tf.logging.info('Done!')
    tf.logging.info('#Num eval samples %d' % len(result))
    return res_file, quest_ids


def main(_):
    from vqa_eval import evaluate_model, write_result_log
    from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file, quest_ids = test()
        print(res_file)
        acc, details = evaluate_model(res_file, quest_ids,
                                      version=FLAGS.version)
        write_result_log(model_path, FLAGS.model_type, acc,
                         details)
        return acc

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset,
                                       FLAGS.model_type)
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    from vqa_eval import evaluate_model

    with tf.Graph().as_default():
        res_file, quest_ids = test()
    acc, details = evaluate_model(res_file, quest_ids,
                                  version=FLAGS.version)
    print('Overall: %0.3f' % acc)
    # tf.app.run()
