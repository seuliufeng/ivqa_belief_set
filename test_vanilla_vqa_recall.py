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
from readers.vqa_naive_data_fetcher import AttentionFetcher
from my_vqa_evaluation import eval_recall

TEST_SET = 'kpval'
_K = 20

tf.flags.DEFINE_string("model_type", "VQA-Base",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_kpvqa_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kptrain",
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
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)
    # to_sentence = SentenceGenerator(trainset='trainval')

    results = []

    print('Running inference on split %s...' % TEST_SET)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        generated_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-2]))
        generated_ans[:, -1] = 0
        ans_cand_ids = np.argsort(-generated_ans, axis=1)

        quest_ids = outputs[-2]

        for quest_id, ids in zip(quest_ids, ans_cand_ids):
            answers = []
            for k in range(_K):
                aid = ids[k]
                ans = to_sentence.index_to_top_answer(aid)
                answers.append(ans)
            res_i = {'question_id': int(quest_id), 'answers': answers}
            results.append(res_i)

    eval_recall(results)


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
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    with tf.Graph().as_default():
        model_path = None
        res_file, quest_ids = test(model_path)
    tf.app.run()
