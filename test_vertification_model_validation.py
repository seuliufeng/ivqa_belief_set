from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

from models.model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from vqa_config import ModelConfig
from readers.vert_base_data_fetcher import TestReader

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "Vert",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_vqa_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_bool("sample_negative", False,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_bool("use_fb_data", False,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_bool("use_fb_bn", True,
                     "Use all answer candidates or just two.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test_worker(model, sess, subset='kpval'):
    # build data reader
    reader = TestReader(batch_size=32, subset=subset,
                        use_fb_data=FLAGS.use_fb_data)

    quest_ids = []
    result = []

    tf.logging.info('\nRunning inference on split %s...' % subset)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        mc_scores = sess.run(
            model._logits, feed_dict=model.fill_feed_dict(outputs[:-3]))
        choice_idx = np.argmax(mc_scores, axis=1)

        cands, _qids, image_ids = outputs[-3:]
        for qid, cid, mcs in zip(_qids, choice_idx, cands):
            answer = mcs['cands'][cid]
            assert (mcs['quest_id'] == qid)
            result.append({u'answer': answer, u'question_id': qid})

        quest_ids.append(_qids)

    return quest_ids, result


def test(checkpoint_path=None):
    subsets = ['kpval', 'kptest', 'kprestval']

    quest_ids = []
    result = []

    config = ModelConfig()
    config.sample_negative = FLAGS.sample_negative
    config.use_fb_bn = FLAGS.use_fb_bn
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build and restore model
    model = model_fn(config, phase='test')
    model.build()

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    for subset in subsets:
        _quest_ids, _result = test_worker(model, sess, subset)
        quest_ids += _quest_ids
        result += _result

    quest_ids = np.concatenate(quest_ids)
    # save results
    tf.logging.info('Saving results')
    res_file = FLAGS.result_format % (FLAGS.version, 'val')
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

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version,
                                       FLAGS.model_type)
    if FLAGS.sample_negative:
        ckpt_dir += '_sn'

    if FLAGS.use_fb_data:
        ckpt_dir += '_fb'

    if FLAGS.use_fb_bn:
        ckpt_dir += '_bn'

    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    tf.app.run()
