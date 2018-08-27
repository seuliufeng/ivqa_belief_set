from __future__ import division
import tensorflow as tf
import numpy as np
import json
import os
from util import update_progress

from models.model_creater import get_model_creation_fn
from inference_utils.question_generator_util import SentenceGenerator
from vqa_config import ModelConfig
from readers.ivqa_reader_creater import create_reader
from rerank_analysis_util import RerankAnalysiser

tf.flags.DEFINE_string("model_type", "VAQ-CA",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v2",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("testset", "test",
                       "Tesint set, val or train.")
tf.flags.DEFINE_string("checkpoint_dir", "/scratch/fl302/inverse_vqa/model/dbg_%s_kpvqa_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("loss_type", "pairwise",
                       "use pairwise loss or softmax loss.")
tf.flags.DEFINE_boolean("restore", True, "restore tuned model or use intitialised one")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(checkpoint_path=None):
    batch_size = 40
    config = ModelConfig()
    config.convert = True
    config.ivqa_rerank = True  # VQA baseline or re-rank
    config.loss_type = FLAGS.loss_type
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)
    # ana_ctx = RerankAnalysiser()

    # build data reader
    reader_fn = create_reader(FLAGS.model_type, phase='test')
    reader = reader_fn(batch_size=batch_size, subset='kp%s' % FLAGS.testset,
                       version=FLAGS.version)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='evaluate')
    model.build()
    # prob = model.prob

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    if FLAGS.restore:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
    else:
        sess.run(tf.initialize_all_variables())
        model.init_fn(sess)

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    ans_ids = []
    quest_ids = []

    b_rerank_scores = []
    b_vqa_scores = []
    b_cand_labels = []
    print('Running inference on split %s...' % FLAGS.testset)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        model_preds = model.inference_rerank_vqa(outputs[:4], sess)
        score, top_ans, _, _, _ = model_preds
        ivqa_score, ivqa_top_ans, ivqa_scores, vqa_top_ans, vqa_scores = model_preds
        b_rerank_scores.append(ivqa_scores)
        b_vqa_scores.append(vqa_scores)
        b_cand_labels.append(vqa_top_ans)
        # if i > 100:
        #     break
        # ana_ctx.update(outputs, model_preds)

        ans_ids.append(top_ans)
        quest_id = outputs[-2]
        quest_ids.append(quest_id)
    # save preds
    b_rerank_scores = np.concatenate(b_rerank_scores, axis=0)
    b_vqa_scores = np.concatenate(b_vqa_scores, axis=0)
    b_cand_labels = np.concatenate(b_cand_labels, axis=0)
    quest_ids = np.concatenate(quest_ids)
    from util import save_hdf5
    save_hdf5('data/rerank_kptest.h5', {'ivqa': b_rerank_scores,
                                         'vqa': b_vqa_scores,
                                         'cands': b_cand_labels,
                                         'quest_ids': quest_ids})

    # ana_ctx.compute_accuracy()

    ans_ids = np.concatenate(ans_ids)
    result = [{u'answer': to_sentence.index_to_top_answer(aid),
               u'question_id': qid} for aid, qid in zip(ans_ids, quest_ids)]

    # save results
    tf.logging.info('Saving results')
    res_file = FLAGS.result_format % (FLAGS.version, FLAGS.testset)
    json.dump(result, open(res_file, 'w'))
    tf.logging.info('Done!')
    tf.logging.info('#Num eval samples %d' % len(result))
    # ana_ctx.close()
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
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    res_file, quest_ids = test('/scratch/fl302/inverse_vqa/model/dbg_v2_kpvaq_VAQ-CA_pairwise/model.ckpt-594000')
    from vqa_eval import evaluate_model, write_result_log

    acc, details = evaluate_model(res_file, quest_ids,
                                  subset=FLAGS.testset,
                                  version=FLAGS.version)
    # tf.app.run()
