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
tf.flags.DEFINE_string("checkpoint_dir", "model/%s%s_vqa_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_bool("sample_negative", True,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_bool("use_fb_data", True,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_bool("use_fb_bn", True,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_bool("use_ap", False,
                     "Use all answer candidates or just two.")

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 64
    config = ModelConfig()
    config.sample_negative = FLAGS.sample_negative
    config.use_fb_bn = FLAGS.use_fb_bn
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = TestReader(batch_size=batch_size, subset=TEST_SET,
                        use_fb_data=FLAGS.use_fb_data)
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

    quest_ids = []
    result = []

    print('Running inference on split %s...' % TEST_SET)
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

    quest_ids = np.concatenate(quest_ids)

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
            res_file, quest_ids = test(model_path)
        print(res_file)
        acc, details = evaluate_model(res_file, quest_ids,
                                      version=FLAGS.version)
        write_result_log(model_path, FLAGS.model_type, acc,
                         details)
        return acc

    mode = 'ap_' if FLAGS.use_ap else ''
    ckpt_dir = FLAGS.checkpoint_dir % (mode,
                                       FLAGS.version,
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
