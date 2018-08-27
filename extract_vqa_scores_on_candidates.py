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
from models.vqa_base import BaseModel as model_fn
from readers.vqa_naive_cand_data_fetcher import AttentionFetcher

tf.flags.DEFINE_string("model_type", "VQA-BaseNorm",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kprestval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("testset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_boolean("use_var", True,
                        "Use variational VQA or VQA.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 100
    config = ModelConfig()
    # Get model function
    # model_fn = get_model_creation_fn(FLAGS.model_type)
    _model_suffix = 'var_' if FLAGS.use_var else ''

    # build data reader
    reader = AttentionFetcher(batch_size=batch_size,
                              subset=FLAGS.testset,
                              feat_type=config.feat_type,
                              version=FLAGS.version,
                              var_suffix=_model_suffix)
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
    # top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    # to_sentence = SentenceGenerator(trainset='trainval',
    #                                 top_ans_file=top_ans_file)
    # to_sentence = SentenceGenerator(trainset='trainval')

    ans_ids = []
    ans_scores = []
    gt_scores = []
    quest_ids = []

    print('Running inference on split %s...' % FLAGS.testset)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        generated_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-2]))
        _gt_labels = outputs[3]
        _this_batch_size = _gt_labels.size
        _gt_scores = generated_ans[np.arange(_this_batch_size, ), _gt_labels]
        gt_scores.append(_gt_scores)
        generated_ans[:, -1] = 0
        top_ans = np.argmax(generated_ans, axis=1)
        top_scores = np.max(generated_ans, axis=1)

        ans_ids.append(top_ans)
        ans_scores.append(top_scores)
        quest_id = outputs[-2]
        quest_ids.append(quest_id)

    quest_ids = np.concatenate(quest_ids)
    ans_ids = np.concatenate(ans_ids)
    ans_scores = np.concatenate(ans_scores)
    gt_scores = np.concatenate(gt_scores)

    # save results
    tf.logging.info('Saving results')
    # res_file = FLAGS.result_format % (FLAGS.version, FLAGS.testset)
    from util import save_hdf5
    save_hdf5('data4/%sivqa_%s_qa_scores.data' % (_model_suffix, FLAGS.testset),
              {'ext_quest_ids': quest_ids,
               'ext_cand_scores': gt_scores,
               'ext_cand_pred_labels': ans_ids,
               'ext_cand_pred_scores': ans_scores})


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

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset,
                                       FLAGS.model_type)
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    test()
    # with tf.Graph().as_default():
    #     model_path = None
    #     res_file, quest_ids = test(model_path)
    # tf.app.run()
