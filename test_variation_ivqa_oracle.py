from __future__ import division
import tensorflow as tf

import numpy as np
import pylab as plt
import os
from skimage.io import imread
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from eval_vqa_question_oracle import evaluate_oracle
# from write_examples import ExperimentWriter
# from visualise_variation_ivqa_beam_search import convert_to_unique_questions
from post_process_variation_questions import post_process_variation_questions_with_count


END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id
IM_ROOT = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco/'

tf.flags.DEFINE_string("model_type", "VAQ-Var",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_var_kptrain_%s",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def find_unique_rows(scores, pathes):
    sorted_data = pathes[np.lexsort(pathes.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    pathes = sorted_data[row_mask]
    scores = np.zeros_like(pathes, dtype=np.float32)
    return scores, pathes


def put_to_array(sentences):
    sentence_lengths = [len(s) for s in sentences]
    max_length = max(sentence_lengths)
    batch_size = len(sentences)
    token_arrays = np.zeros([batch_size, max_length], dtype=np.int32)
    for s, s_len, target in zip(sentences, sentence_lengths, token_arrays):
        target[:s_len] = s
    token_lens = np.array(sentence_lengths, dtype=np.int32)
    return token_arrays.astype(np.int32), token_lens


def convert_to_unique_questions(scores, pathes):
    scores, pathes = post_process_prediction(scores, pathes)
    pathes, pathes_len = put_to_array(pathes)
    scores, pathes = find_unique_rows(scores, pathes)
    return post_process_prediction(scores, pathes[:, 1:])


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


def extract_gt(capt, capt_len):
    gt = []
    for c, c_len in zip(capt, capt_len):
        tmp = c[:c_len].tolist()
        gt.append(np.array(tmp))
    return gt


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


def ivqa_decoding_beam_search(checkpoint_path=None, subset='kptest'):
    model_config = ModelConfig()
    res_file = 'result/aug_var_vaq_kl0_greedy_%s.json' % FLAGS.model_type.upper()
    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader('VAQ-Var', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=1, subset=subset,
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
        model = model_fn(model_config, 'sampling_beam')
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')
    results = []
    for i in range(num_batches):
        print('iter: %d/%d' % (i, num_batches))
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        scores, pathes = model.greedy_inference(outputs[:-2], sess)
        scores = np.tile(scores[:, np.newaxis], [1, pathes.shape[1]])
        # scores, pathes = post_process_prediction(scores, pathes)

        _ntot = len(pathes)
        scores, pathes, ivqa_counts = post_process_variation_questions_with_count(scores, pathes, 1)

        question_id = int(quest_ids[0])
        image_id = image_ids[0]

        print('%d/%d' % (len(pathes[0]), _ntot))
        for _p_idx, (path, sc) in enumerate(zip(pathes[0], scores[0])):
            sentence = to_sentence.index_to_question(path)
            aug_quest_id = question_id * 1000 + _p_idx
            # res_i = {'image_id': int(image_id),
            #          'question_id': aug_quest_id,
            #          'question': sentence}
            res_i = {'image_id': int(image_id),
                     'question_id': aug_quest_id,
                     'question': sentence,
                     'question_inds': path,
                     'counts': len(pathes),
                     'probs': float(sc)}
            results.append(res_i)

    save_json(res_file, results)
    return res_file


def main(_):
    from watch_model import ModelWatcher
    subset = 'kptest'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = ivqa_decoding_beam_search(checkpoint_path=model_path,
                                                 subset=subset)
            cider = evaluate_oracle(res_file)
        return cider

    #  test_model(None)
    #  exit(0)

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    ckpt_dir = 'model/v1_var_ivqa_restval_VAQ-Var/model.ckpt-560000'
    test_model(ckpt_dir)
    # res_file = ivqa_decoding_beam_search(None,
    #                                      subset=subset)
    # print(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    tf.app.run()
