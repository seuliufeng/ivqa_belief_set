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

tf.flags.DEFINE_string("model_type", "VAQ-SAT",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "/scratch/fl302/inverse_vqa/model/%s_kpvaq_%s",
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


def parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen])
    return seqs


def ivqa_decoding_beam_search(checkpoint_path=None, subset='kpval'):
    model_config = ModelConfig()
    res_file = 'result/quest_vaq_greedy_%s.json' % FLAGS.model_type.upper()
    # Get model
    # model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader(FLAGS.model_type, phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval')

    # get data reader
    reader = create_fn(batch_size=80, subset=subset,
                       version=FLAGS.test_version)

    # if checkpoint_path is None:
    #     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir %
    #                                          (FLAGS.version,
    #                                           FLAGS.model_type))
    #     checkpoint_path = ckpt.model_checkpoint_path

    # Build model
    # g = tf.Graph()
    # with g.as_default():
    #     # Build the model.
    #     model = model_fn(model_config, 'beam')
    #     model.build()
    #     # Restore from checkpoint
    #     restorer = Restorer(g)
    #     sess = tf.Session()
    #     restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')
    results = []
    for i in range(num_batches):
        outputs = reader.get_test_batch()

        # inference
        # quest_ids, image_ids = outputs[-2:]
        # scores, pathes = model.greedy_inference(outputs[:-2], sess)
        im, capt, capt_len, ans_seq, ans_seq_len, quest_ids, image_ids = outputs

        _, res, res_len, _, _, _, _, = reader.get_test_batch()

        pathes = parse_gt_questions(capt, capt_len)
        question = to_sentence.index_to_question(pathes[0])
        gts = [to_sentence.index_to_question(q) for q in pathes]
        gts_token = [' '.join([str(t) for t in path]) for path in pathes]

        respathes = parse_gt_questions(res, res_len)
        res = [to_sentence.index_to_question(q) for q in respathes]
        res_token = [' '.join([str(t) for t in path]) for path in respathes]

        scores = compute_cider_token_1vsall(quest_ids, res_token)
        import pdb
        pdb.set_trace()

        # gts_token = []
        # # compute_cider(quest_ids, gts, res)
        # compute_cider_token(quest_ids, gts_token, res_token)
        # import pdb
        # pdb.set_trace()
        # print('%d/%d: %s' % (i, num_batches, question))
        #
        # for quest_id, image_id, path in zip(quest_ids, image_ids, pathes):
        #     sentence = to_sentence.index_to_question(path)
        #     res_i = {'image_id': int(image_id), 'question_id': int(quest_id), 'question': sentence}
        #     results.append(res_i)

    save_json(res_file, results)
    return res_file


def compute_cider(quest_ids, gts, res):
    # warp gts
    w_gts = {str(quest_id): [{'caption': capt}] for quest_id, capt in zip(quest_ids, gts)}
    w_res = [{'image_id': str(quest_id), 'caption': capt} for quest_id, capt in zip(quest_ids, res)]
    from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
    scorer = ciderEval(w_gts, w_res, 'ivqa_train_words')
    scores = scorer.evaluate()
    return scores


def compute_cider_token(quest_ids, gts, res):
    # warp gts
    w_gts = {str(quest_id): [capt] for quest_id, capt in zip(quest_ids, gts)}
    w_res = [{'image_id': str(quest_id), 'caption': [capt]} for quest_id, capt in zip(quest_ids, res)]
    from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
    scorer = ciderEval(w_gts, w_res, 'ivqa_train_idxs')
    scores = scorer.evaluate()
    return scores


def compute_cider_token_1vsall(quest_ids, res):
    # warp gts
    w_gts = {str(quest_id): res for quest_id in quest_ids}
    w_res = [{'image_id': str(quest_id), 'caption': [capt]} for quest_id, capt in zip(quest_ids, res)]
    from pyciderevalcap.fast_eval import CIDErEvalCap as ciderEval
    from time import time
    # t = time()
    scorer = ciderEval(w_gts, w_res, 'ivqa_train_idxs')
    scores = scorer.evaluate()
    # print(time()-t)
    return scores


def main(_):
    from watch_model import ModelWatcher
    subset = 'kpval'

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = ivqa_decoding_beam_search(subset=subset)
            cider = evaluate_question(res_file, subset=subset,
                                      version=FLAGS.test_version)
        return cider

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    ivqa_decoding_beam_search()
    # tf.app.run()
