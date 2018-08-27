from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

from vqa_model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from config import ModelConfig
from attention_data_fetcher import AttentionFetcher
from prediction_analysizer import PredictionVisualiser
# from inception_attention_data_fetcher import AttentionFetcher
from w2v_answer_encoder import MultiChoiceQuestionManger

# TEST_SET = 'test-dev'
TEST_SET = 'dev'

tf.flags.DEFINE_string("model_type", "VQA-Incept",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/curr_%s_%s",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_file", "result/vqa_OpenEnded_mscoco_%s_baseline_results.json" % TEST_SET,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 1
    config = ModelConfig()
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = AttentionFetcher(batch_size=batch_size, subset=TEST_SET)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir %
                                             (FLAGS.model_type, config.feat_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    model.build()
    g_prob = model.prob
    g_att_map = model.attention_map
    # sess = tf.Session()
    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Create the vocabulary.
    visualiser = PredictionVisualiser(FLAGS.model_type, do_plot=True)

    ans_ids = []
    quest_ids = []
    print('Running inference on split %s...' % TEST_SET)
    for i in range(reader.num_batches):
        if i % 100 == 0:
            update_progress(i/float(reader.num_batches))
        outputs = reader.get_test_batch()
        if i < 100:
            continue
        generated_ans, att_map = sess.run(
            [g_prob, g_att_map], feed_dict=model.fill_feed_dict(outputs[:-2]))
        # process attention map
        att_map = att_map.reshape([batch_size, 14, 14, -1])
        att_map = np.transpose(att_map, [0, 3, 1, 2])
        generated_ans[:, -1] = 0
        top_ans = np.argmax(generated_ans, axis=1)
        gt_ans = outputs[3]

        ans_ids.append(top_ans)
        quest_id = outputs[-2]
        quest_ids.append(quest_id)

        if np.random.rand() > 0.05:
            visualiser.plot(quest_id, generated_ans, att_map)

        # qid = quest_id[-1]
        # aid = top_ans[-1]
        # qs = quest_manager.get_question(qid)
        # # sample a question to visualise
        # ans = to_sentence.index_to_top_answer(aid)
        # print('============== %d ============' % (i*batch_size))
        # print('question id: %d' % qid)
        # print('Q: %s' % qs.title())
        # print('A: %s\n' % ans.title())
    # quest_ids = np.concatenate(quest_ids)
    # ans_ids = np.concatenate(ans_ids)
    # result = [{u'answer': to_sentence.index_to_top_answer(aid),
    #            u'question_id': qid} for aid, qid in zip(ans_ids, quest_ids)]
    # save results
    # tf.logging.info('Saving results')
    # res_file = FLAGS.result_file
    # json.dump(result, open(res_file, 'w'))
    # tf.logging.info('Done!')
    # return res_file, quest_ids


def test_on_test_dev():
    ckpt_path = 'model/res_VQA-G/model.ckpt-1000'
    with tf.Graph().as_default():
        res_file, quest_ids = test(ckpt_path)


def main(_):
    from vqa_eval import evaluate_model
    test()


if __name__ == '__main__':
    # test_on_test_dev()
    # test(checkpoint_path=None)
    tf.app.run()
