from __future__ import division
import tensorflow as tf

import numpy as np
from readers.ivqa_reader_creater import create_reader
from models.contrastive_question_sampler import ContrastQuestionSampler
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id

tf.flags.DEFINE_string("model_type", "VAQ-CA",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v2",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v2",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "/scratch/fl302/inverse_vqa/model/%s_kpvaq_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("convert", False, "convert checkpoint from a different version")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_question(result_file, subset='kptest', version='v1'):
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


def sample_cst_questions(checkpoint_path=None, subset='kptrain'):
    model_config = ModelConfig()
    model_config.convert = FLAGS.convert
    model_config.loss_type = 'pairwise'
    model_config.top_k = 3
    batch_size = 8
    # Get model
    create_fn = create_reader(FLAGS.model_type, phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=batch_size, subset=subset,
                       version=FLAGS.test_version)

    # Build model
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = ContrastQuestionSampler(model_config)
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')

    for i in range(num_batches):
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        c_ans, c_ans_len, pathes, scores = model.greedy_inference(outputs[:-2], sess)
        scores, pathes = post_process_prediction(scores, pathes)

        k = 3
        capt, capt_len = outputs[2: 4]

        gt = capt[0, :capt_len[0]]
        print('gt: %s [%s]' % (to_sentence.index_to_question(gt),
                               to_sentence.index_to_answer(c_ans[0, :c_ans_len[0]])))
        for ix in range(k):
            question = to_sentence.index_to_question(pathes[ix])
            answer = to_sentence.index_to_answer(c_ans[ix, :c_ans_len[ix]])
            print('%s %d: %s [%s]' % ('pre' if ix == 0 else 'cst', ix, question, answer))
        import pdb
        pdb.set_trace()


def main(_):
    ckpt_path = '/scratch/fl302/inverse_vqa/model/dbg_v2_kpvaq_VAQ-CA_pairwise/model.ckpt-0'
    sample_cst_questions(ckpt_path, 'kptrain')


if __name__ == '__main__':
    tf.app.run()
