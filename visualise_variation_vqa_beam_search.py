from __future__ import division
import tensorflow as tf

import numpy as np
import pylab as plt
import os
from skimage.io import imread
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, ANS_VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from write_examples import ExperimentWriter
import pdb

END_TOKEN = ANS_VOCAB_CONFIG.end_token_id
START_TOKEN = ANS_VOCAB_CONFIG.start_token_id
IM_ROOT = '/usr/data/fl302/data/VQA/Images/'

tf.flags.DEFINE_string("model_type", "VQA-Var",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_var_vqa_trainval_%s",
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


def sample_unique_questions(scores, pathes):
    scores, pathes = post_process_prediction(scores, pathes)
    pathes, pathes_len = put_to_array(pathes)
    scores, pathes = find_unique_rows(scores, pathes)
    scores, pathes = post_process_prediction(scores, pathes[:, 1:])


def var_vqa_decoding_beam_search(checkpoint_path=None, subset='kpval'):
    model_config = ModelConfig()
    res_file = 'result/quest_vaq_greedy_%s.json' % FLAGS.model_type.upper()
    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader(FLAGS.model_type, phase='test')
    writer = ExperimentWriter('latex/examples_%s' % FLAGS.model_type.lower())

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    subset = 'kprestval'
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
        model = model_fn(model_config, 'sampling')
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
        # pdb.set_trace()

        # inference
        images, quest, quest_len, ans, ans_len, quest_ids, image_ids = outputs
        scores, pathes = model.greedy_inference([images, quest, quest_len], sess)
        scores, pathes = post_process_prediction(scores, pathes)
        pathes, pathes_len = put_to_array(pathes)
        scores, pathes = find_unique_rows(scores, pathes)
        scores, pathes = post_process_prediction(scores, pathes[:, 1:])
        # question = to_sentence.index_to_question(pathes[0])
        # print('%d/%d: %s' % (i, num_batches, question))

        # show image
        os.system('clear')
        im_file = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_ids[0])
        im_path = os.path.join(IM_ROOT, im_file)
        # im = imread(im_path)
        # plt.imshow(im)
        questions = extract_gt(quest, quest_len)
        question = to_sentence.index_to_question(questions[0])
        print('Question: %s' % question)

        answers = extract_gt(ans, ans_len)
        answer = to_sentence.index_to_answer(answers[0])
        # plt.title(answer)

        print('Answer: %s' % answer)
        answers = []
        for path in pathes:
            sentence = to_sentence.index_to_answer(path)
            answers.append(sentence)
            print(sentence)
        # plt.show()
        qa = '%s - %s' % (question, answer)
        writer.add_result(image_ids[0], quest_ids[0], im_path, qa, answers)

        for quest_id, image_id, path in zip(quest_ids, image_ids, pathes):
            sentence = to_sentence.index_to_question(path)
            res_i = {'image_id': int(image_id), 'question_id': int(quest_id), 'question': sentence}
            results.append(res_i)

        if i == 40:
            break

    writer.render()
    return


def main(_):
    from watch_model import ModelWatcher
    subset = 'kpval'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = var_vqa_decoding_beam_search(checkpoint_path=model_path,
                                                    subset=subset)
            cider = evaluate_question(res_file, subset=subset,
                                      version=FLAGS.test_version)
        return cider

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL_ft/'
    res_file = var_vqa_decoding_beam_search(None,
                                            subset=subset)
    # print(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    tf.app.run()
