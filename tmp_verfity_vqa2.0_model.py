from __future__ import division
import tensorflow as tf
from rerank_fusion import RerankModel, Reader
import os
import numpy as np
import json
import pdb
from util import update_progress
from inference_utils.question_generator_util import SentenceGenerator

tf.flags.DEFINE_string("checkpoint_dir", "/import/vision-ephemeral/fl302/data/model/%s_kpvaq2_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 10000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test():
    from util import load_hdf5
    d = load_hdf5('data/rerank_kpval.h5')
    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=None)
    quest_ids = d['quest_ids']
    ans_ids = d['cands'][:, 0]
    # vqa_scores = d['vqa']

    result = [{u'answer': to_sentence.index_to_top_answer(aid),
               u'question_id': qid} for aid, qid in zip(ans_ids, quest_ids)]

    # save results
    tf.logging.info('Saving results')
    res_file = FLAGS.result_format % ('v2', 'kpval')
    json.dump(result, open(res_file, 'w'))
    tf.logging.info('Done!')
    tf.logging.info('#Num eval samples %d' % len(result))
    # ana_ctx.close()
    return res_file, quest_ids


def main():
    from vqa_eval import evaluate_model, write_result_log
    from watch_model import ModelWatcher

    # def test_model(model_path):
    #     with tf.Graph().as_default():
    res_file, quest_ids = test()
    print(res_file)
    acc, details = evaluate_model(res_file, quest_ids,
                                  version='v2')
    # write_result_log(model_path, 'Fusion', acc,
    #                  details)
    # return acc

    # ckpt_dir = FLAGS.checkpoint_dir % ('v2',
    #                                    'Fusion')
    # # print(ckpt_dir)
    # # test_model(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    main()
