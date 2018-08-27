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


def test(checkpoint_path=None):
    batch_size = 128

    # build data reader
    reader = Reader(batch_size=batch_size, subset='kpval', phase='test')

    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % ('v2',
                                                                     'Fusion'))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = RerankModel(phase='test')
    model.build()

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    ans_ids = []
    quest_ids = []

    print('Running inference on split %s...' % 'kpval')
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.pop_batch()
        gates, model_preds, normed = sess.run([model.gates, model.preds, model._norm_ivqa_scores],
                                              feed_dict=model.fill_feed_dict(outputs))
        local_index = model_preds.argmax(axis=1)
        # local_index = outputs[-3].argmax(axis=1)
        top_ans = np.array([cand[idx] for idx, cand in zip(local_index, outputs[3])])

        ans_ids.append(top_ans)
        quest_id = outputs[-1]
        quest_ids.append(quest_id)

    ans_ids = np.concatenate(ans_ids)
    quest_ids = np.concatenate(quest_ids)
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

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file, quest_ids = test(model_path)
        print(res_file)
        acc, details = evaluate_model(res_file, quest_ids,
                                      version='v2')
        write_result_log(model_path, 'Fusion', acc,
                         details)
        return acc

    ckpt_dir = FLAGS.checkpoint_dir % ('v2',
                                       'Fusion')
    # print(ckpt_dir)
    # test_model(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    main()
