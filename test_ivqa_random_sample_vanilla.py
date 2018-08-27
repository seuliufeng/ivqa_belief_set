from __future__ import division
import tensorflow as tf

import numpy as np
from util import save_json
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from eval_vqa_question_oracle import evaluate_oracle
from post_process_variation_questions import post_process_variation_questions_with_count
import pdb

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id
IM_ROOT = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco/'

tf.flags.DEFINE_string("model_type", "iVQA-Basic",
                       "Select a model to train.")
tf.flags.DEFINE_string("mode", "full",
                       "How to select candidates, max_count, max_score, or full")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("model_trainset", "kprestval",
                       "Which split is the model trained on")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_question_standard(result_file, subset='kptest', version='v1'):
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


def pick_question(scores_i, pathes_i, counts_i):
    if FLAGS.mode == 'max_count':
        idx = np.argmax(counts_i)
    elif FLAGS.mode == 'max_score':
        idx = np.argmax(scores_i)
    else:
        raise Exception('illegal option')
    return pathes_i[idx]


def ivqa_decoding_beam_search(checkpoint_path=None, subset='kptest'):
    model_config = ModelConfig()
    res_file = 'result/var_vaq_rand_%s_%s.json' % (FLAGS.model_type.upper(), FLAGS.mode)
    # return res_file
    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader(FLAGS.model_type, phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=10, subset=subset,
                       version=FLAGS.test_version)

    if checkpoint_path is None:
        ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset, FLAGS.model_type)
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
    ind_counts = []
    for i in range(num_batches):
        print('iter: %d/%d' % (i, num_batches))
        # if i >= 10:
        #     break
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        pathes, scores = model.greedy_inference(outputs[:-2], sess)

        # wrap inputs
        _this_batch_size = quest_ids.size
        seq_len = pathes.shape[1]
        dummy_scores = scores
        # dummy_scores = np.tile(scores[:, np.newaxis], [1, seq_len])
        # dummy_scores = np.zeros_like(pathes, dtype=np.float32)
        ivqa_scores, ivqa_pathes, ivqa_counts = post_process_variation_questions_with_count(dummy_scores, pathes,
                                                                                            _this_batch_size)
        # scores, pathes = convert_to_unique_questions(scores, pathes)

        for _q_idx, (ps, scs, cs) in enumerate(zip(ivqa_pathes, ivqa_scores, ivqa_counts)):
            image_id = image_ids[_q_idx]
            question_id = int(quest_ids[_q_idx])
            ind_counts.append(len(cs))
            # if not _q_idx:
            #     print('\n%d' % question_id)
            if FLAGS.mode == 'full':
                for _p_idx, (p, sc, _c) in enumerate(zip(ps, scs, cs)):
                    sentence = to_sentence.index_to_question(p)
                    aug_quest_id = question_id * 1000 + _p_idx
                    res_i = {'image_id': int(image_id),
                             'question_id': aug_quest_id,
                             'question': sentence,
                             'question_inds': p,
                             'counts': _c,
                             'probs': float(sc)}
                    # res_i = {}
                    # if not _q_idx:
                    #     print(sentence)
                    results.append(res_i)
            else:
                p = pick_question(scs, ps, cs)
                sentence = to_sentence.index_to_question(p)
                # print(sentence)
                res_i = {'image_id': int(image_id),
                         'question_id': question_id,
                         'question': sentence}
                # res_i = {}
                results.append(res_i)
        # pdb.set_trace()

    save_json(res_file, results)
    print('\nAverage counts: %0.2f' % (np.mean(ind_counts)))
    return res_file


def main(_):
    from watch_model import ModelWatcher
    subset = 'kptest'
    # subset = 'kpval'
    # subset = 'kptrain'
    target_split = 'train' if 'train' in subset else 'val'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = ivqa_decoding_beam_search(checkpoint_path=model_path,
                                                 subset=subset)
            if FLAGS.mode == 'full':
                cider = evaluate_oracle(res_file, split=target_split)
            else:
                cider = evaluate_question_standard(res_file)
        return float(cider)

    #
    # ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.model_trainset, FLAGS.model_type)
    test_model(None)
    # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL_ft/'
    # res_file = ivqa_decoding_beam_search(None,
    #                                     subset=subset)
    # print(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    tf.app.run()
