from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

from models.model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from vqa_config import ModelConfig
from readers.vqa_naive_data_fetcher import AttentionFetcher

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "VQA-Adv",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kprestval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_%s_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 100
    config = ModelConfig()
    # Get model function
    model_fn = get_model_creation_fn('VQA-Base')

    # build data reader
    reader = AttentionFetcher(batch_size=batch_size, subset=TEST_SET,
                              feat_type=config.feat_type, version=FLAGS.version)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    model.build()
    prob = model.prob

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Create the vocabulary.
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)

    ans_ids = []
    quest_ids = []

    print('Running inference on split %s...' % TEST_SET)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        generated_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-2]))
        generated_ans[:, -1] = 0
        top_ans = np.argmax(generated_ans, axis=1)

        ans_ids.append(top_ans)
        quest_id = outputs[-2]
        quest_ids.append(quest_id)

    quest_ids = np.concatenate(quest_ids)
    ans_ids = np.concatenate(ans_ids)
    result = [{u'answer': to_sentence.index_to_top_answer(aid),
               u'question_id': qid} for aid, qid in zip(ans_ids, quest_ids)]

    # save results
    tf.logging.info('Saving results')
    res_file = FLAGS.result_format % (FLAGS.version, TEST_SET,
                                      FLAGS.model_type)
    json.dump(result, open(res_file, 'w'))
    tf.logging.info('Done!')
    tf.logging.info('#Num eval samples %d' % len(result))
    return res_file, quest_ids


def main(_):
    from vqa_eval import evaluate_model, write_result_log
    from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file, quest_ids = test(model_path)
        print(res_file)
        acc, details = evaluate_model(res_file, quest_ids,
                                      version=FLAGS.version)
        write_result_log(model_path, FLAGS.model_type, acc,
                         details)
        return acc

    # ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version,
    #                                    FLAGS.model_type)
    # ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset,
    #                                    FLAGS.model_type)
    ckpt_dir = 'model/vqa_adv/'
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    tf.app.run()
