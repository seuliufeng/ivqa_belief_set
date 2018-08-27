from __future__ import division
import tensorflow as tf

import os
import numpy as np
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.vqa_model_creater import get_model_creation_fn
from config import ModelConfig
import vaq_utils
from inference_utils.question_generator_util import SentenceGenerator

END_TOKEN = vaq_utils._END_TOKEN_ID
START_TOKEN = vaq_utils._START_TOKEN_ID

tf.flags.DEFINE_string("model_type", "VQG",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/kpvaq_%s",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_question(result_file, subset='kptest'):
    from analysis.eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()
    # assert (subset in ['train', 'dev', 'val'])
    subset = 'train' if subset == 'train' else 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    return evaluator.get_overall_blue4()


def post_process_prediction(scores, pathes):
    pred_len = np.argmax(np.equal(pathes, END_TOKEN), axis=1)
    confs, vpathes = [], []
    for path, score, plen in zip(pathes, scores, pred_len):
        conf = np.sum(score[:plen])
        seq = path.tolist()[:plen]
        seq = [START_TOKEN] + seq + [END_TOKEN]
        confs.append(conf)
        vpathes.append(seq)
    return confs, vpathes


def vaq_decoding_greedy(checkpoint_path=None, subset='kpval'):
    model_config = ModelConfig()
    res_file = 'result/quest_vaq_greedy_%s.json' % FLAGS.model_type.upper()

    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader(FLAGS.model_type, phase='test')
    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # build data reader
    reader = create_fn(batch_size=32, subset=subset)

    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'greedy')
        model.build()
        saver = tf.train.Saver()

        sess = tf.Session()
        tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
        saver.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running greedy inference...')
    results = []
    for i in range(num_batches):
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        scores, pathes = model.greedy_inference(outputs[:-2], sess)

        scores, pathes = post_process_prediction(scores, pathes)
        question = to_sentence.index_to_question(pathes[0])
        print('%d/%d: %s' % (i, num_batches, question))

        for quest_id, image_id, path in zip(quest_ids, image_ids, pathes):
            sentence = to_sentence.index_to_question(path)
            res_i = {'image_id': int(image_id), 'question_id': int(quest_id), 'question': sentence}
            results.append(res_i)

    save_json(res_file, results)
    return res_file


def main(_):
    from watch_model import ModelWatcher
    subset = 'kpval'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = vaq_decoding_greedy(subset=subset)
            cider = evaluate_question(res_file, subset=subset)
        return cider

    ckpt_dir = FLAGS.checkpoint_dir % FLAGS.model_type
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    tf.app.run()
