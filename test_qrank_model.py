from __future__ import division
import tensorflow as tf
import numpy as np
import os
from util import update_progress

from models.model_creater import get_model_creation_fn
from config import ModelConfig
from readers.question_ranking_fetcher import TestReader

# from test_models_beam_search import evaluate_question

TEST_SET = 'kptest'

tf.flags.DEFINE_string("model_type", "QRank",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_qrank_%s_%g",
                       "Model checkpoint file.")
tf.flags.DEFINE_bool("test_once", True,
                     "Use all answer candidates or just two.")
tf.flags.DEFINE_float("delta", 0.5,
                      "CIDEr margin for build contrastive pairs")

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_question(result_file, subset='kpval', version='v1'):
    from analysis.eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()

    subset = 'train' if subset == 'train' else 'val'
    if version == 'v1':
        annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
        question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)
    elif version == 'v2':
        anno_dir = '/import/vision-ephemeral/fl302/data/VQA2.0'
        annotation_file = '%s/v2_mscoco_%s2014_annotations.json' % (anno_dir, subset)
        question_file = '%s/v2_OpenEnded_mscoco_%s2014_questions.json' % (anno_dir, subset)
    else:
        raise Exception('unknown version, v1 or v2')

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    # return evaluator.get_overall_blue4()
    return evaluator.get_overall_cider()


def test(checkpoint_path=None):
    batch_size = 128
    config = ModelConfig()
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = TestReader(batch_size=batch_size, subset=TEST_SET)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type,
                                                                     FLAGS.delta))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    model.build()

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    print('Running inference on split %s...' % TEST_SET)
    aug_quest_ids, scores = [], []
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        rank_score = sess.run(
            model.prob, feed_dict=model.fill_feed_dict(outputs[:3]))

        _, quest_ids, image_ids = outputs[3:]
        scores.append(rank_score)
        aug_quest_ids.append(quest_ids)

    aug_quest_ids = np.concatenate(aug_quest_ids)
    scores = np.concatenate(scores)
    return convert_to_questions(aug_quest_ids, scores)


def convert_to_questions(aug_quest_ids, scores):
    cand_file = 'result/var_vaq_beam_VAQ-VAR_full_%s.json' % TEST_SET
    from util import load_json, save_json
    cands = load_json(cand_file)

    # parse cands
    cands = {res['question_id']: {'question': res['question'],
                                  'image_id': int(res['image_id'])} for res in cands}

    # parse
    result = {}
    for aug_id, sc in zip(aug_quest_ids, scores):
        quest_id = int(aug_id / 1000)
        item = [aug_id, sc]
        if quest_id not in result:
            result[quest_id] = [item]
        else:
            result[quest_id].append(item)

    # select question
    results = []
    for quest_id, res in result.items():
        arr = np.array(res)
        aids = arr[:, 0]
        scs = arr[:, 1]
        pick_idx = scs.argmax()
        pick_aug_id = int(aids[pick_idx])
        res_item = cands[pick_aug_id]
        res_i = {'image_id': res_item['image_id'],
                 'question_id': int(quest_id),
                 'question': res_item['question']}
        results.append(res_i)
    sv_file = cand_file + '.rerank'
    save_json(sv_file, results)
    return sv_file


def main(_):
    from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            acc = test_once(model_path)
        return acc

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version,
                                       FLAGS.model_type,
                                       FLAGS.delta)

    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


def test_once(checkpoint_path=None):
    if checkpoint_path is None:
        ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version,
                                           FLAGS.model_type,
                                           FLAGS.delta)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        checkpoint_path = ckpt.model_checkpoint_path

    res_file = test(checkpoint_path)
    cider = evaluate_question(res_file, TEST_SET,
                              version=FLAGS.version)
    print('Overall: %0.2f' % cider)
    return float(cider)


if __name__ == '__main__':
    if FLAGS.test_once:
        test_once()
    else:
        tf.app.run()
