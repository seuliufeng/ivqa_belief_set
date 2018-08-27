from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

# from models.model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from vqa_config import ModelConfig
# from naive_ensemble_model import NaiveEnsembleModel as model_fn
from readers.vqa_irrelevance_data_fetcher import AttentionFetcher
from models.vqa_soft_attention_v2_with_qrd import AttentionModel as model_fn

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "VQA-QRD-BF-MTL",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_vqa_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kprestval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 100
    config = ModelConfig()
    # Get model function
    # model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = AttentionFetcher(batch_size=batch_size, subset=TEST_SET,
                              feat_type=config.feat_type, version=FLAGS.version)
    if checkpoint_path is None:
        print(FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type))
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    # model.set_agent_ids([0])
    model.build()
    prob = model.qrd_prob

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(graph=tf.get_default_graph(),
                      config=tf.ConfigProto(gpu_options=gpu_options))
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    gts = []
    preds = []

    print('Running inference on split %s...' % TEST_SET)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        scores = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs))
        preds.append(scores.flatten())
        gts.append(outputs[-1])

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    from scipy.io import savemat
    from sklearn.metrics import average_precision_score
    sv_file_name = os.path.basename(checkpoint_path)
    savemat('result/predictions_%s.mat' % sv_file_name, {'gt': gts,
                                                         'preds': preds})
    ap = average_precision_score(1.0 - gts, 1.0 - preds)

    return float(ap)


def main(_):
    from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            ap = test(model_path)
        return ap

    # ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    ckpt_dir = '/data1/fl302/projects/vqa2.0/model/curr_VQA-Soft-QRD_Res5c'
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


def test_once():
    model_path = 'model/v1_vqa_VQA-QRD-BF/v1_vqa_VQA-QRD-BF_best2/model.ckpt-70000'
    ap = test(model_path)
    print(ap)


if __name__ == '__main__':
    # with tf.Graph().as_default():
    #     model_path = None
    #     res_file, quest_ids = test(model_path)
    # test_once()
    tf.app.run()
    # test()
