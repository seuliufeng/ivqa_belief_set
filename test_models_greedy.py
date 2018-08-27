from __future__ import division
import tensorflow as tf

import numpy as np
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id

tf.flags.DEFINE_string("model_type", "VAQ-2Att",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v2",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v2",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "/scratch/fl302/inverse_vqa/model/%s_kpvaq2_%s",
                       "Directory for saving and loading model checkpoints.")
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


def post_process_prediction(scores, pathes):
    is_end_token = np.equal(pathes, END_TOKEN)
    pred_len = np.argmax(is_end_token, axis=1)
    exceed_max_len = np.logical_not(np.any(is_end_token, axis=1))
    pred_len[exceed_max_len] = pathes.shape[1]

    confs, vpathes = [], []
    for path, score, plen in zip(pathes, scores, pred_len):
        conf = score
        seq = path.tolist()[:plen]
        seq = [START_TOKEN] + seq + [END_TOKEN]
        confs.append(conf)
        vpathes.append(seq)
    return confs, vpathes


def ivqa_decoding_beam_search(checkpoint_path=None, subset='kpval'):
    model_config = ModelConfig()
    res_file = 'result/quest_vaq_greedy_%s.json' % FLAGS.model_type.upper()
    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader(FLAGS.model_type, phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=100, subset=subset,
                       version=FLAGS.test_version)

    if checkpoint_path is None:
        ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
        # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL/'
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        checkpoint_path = ckpt.model_checkpoint_path

    # Build model
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'greedy')
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')
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
            res_file = ivqa_decoding_beam_search(checkpoint_path=model_path,
                                                 subset=subset)
            cider = evaluate_question(res_file, subset=subset,
                                      version=FLAGS.test_version)
        return cider

    # ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-Mixer_ft'
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    tf.app.run()
