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
from readers.va_naive_data_fetcher import AttentionFetcher as Reader
# from readers.semi_naive_data_fetcher import SemiReader as Reader
# from naive_ensemble_model import NaiveEnsembleModel as model_fn
from models.va_base import BaseModel as model_fn
import pdb

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "VA-Base",
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


def test(checkpoint_path=None):
    batch_size = 100
    config = ModelConfig()
    # Get model function
    # model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = Reader(batch_size=batch_size,
                    subset=TEST_SET,
                    feat_type=config.feat_type,
                    version=FLAGS.version)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    # model.set_agent_ids([0])
    model.build()
    prob = model.prob

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Create the vocabulary.
    quest_ids = []
    ans_preds = []
    gt_labels = []

    print('Running inference on split %s...' % TEST_SET)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        generated_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-2]))
        _gt_labels = outputs[1]
        gt_labels.append(_gt_labels)
        ans_preds.append(generated_ans)

        quest_id = outputs[-2]
        quest_ids.append(quest_id)

    ans_preds = np.concatenate(ans_preds)
    gt_labels = np.concatenate(gt_labels)
    return evaluate_result(ans_preds, gt_labels)


def evaluate_result(preds, labels):
    from scipy.io import savemat, loadmat
    savemat('result/tmp.mat', {'labels': labels,
                               'scores': preds})
    eval_cmd = 'matlab -nojvm -nodisplay -r "eval_word_detection;exit;"'
    print('\n')
    print(eval_cmd)
    os.system(eval_cmd)
    d = loadmat('result/res.mat')
    ap = d['apl'].flatten()[0]
    return ap * 100.


def main(_):
    from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            ap = test(model_path)
        return float(ap)

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset,
                                       FLAGS.model_type)
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    # with tf.Graph().as_default():
    #     model_path = None
    #     res_file, quest_ids = test(model_path)
    tf.app.run()
