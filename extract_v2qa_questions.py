from __future__ import division
import tensorflow as tf

import numpy as np
from util import save_json, save_hdf5
from readers.ivqa_reader_creater import create_reader
from models.model_creater import get_model_creation_fn
from config import ModelConfig, VOCAB_CONFIG
from inference_utils.question_generator_util import SentenceGenerator
from restorer import Restorer
from eval_vqa_question_oracle import evaluate_oracle
import pdb

# from write_examples import ExperimentWriter
# from visualise_variation_ivqa_beam_search import convert_to_unique_questions

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id
IM_ROOT = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco/'

tf.flags.DEFINE_string("model_type", "V2QA-Var",
                       "Select a model to train.")
tf.flags.DEFINE_string("subset", "kptrain",
                       "Dataset for question extraction.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("test_version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_var_v2qa_restval_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("use_var", True,
                        "Use variational VQA or VQA.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

if not FLAGS.use_var:
    FLAGS.model_type = 'iVQA-Basic'
    FLAGS.checkpoint_dir = 'model/%s_%s'


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


def post_process_prediction(scores, pathes, add_start_end=True):
    is_end_token = np.equal(pathes, END_TOKEN)
    pred_len = np.argmax(is_end_token, axis=1)
    exceed_max_len = np.logical_not(np.any(is_end_token, axis=1))
    pred_len[exceed_max_len] = pathes.shape[1]

    confs, vpathes = [], []
    for path, score, plen in zip(pathes, scores, pred_len):
        conf = score
        seq = path.tolist()[:plen]
        if add_start_end:
            seq = [START_TOKEN] + seq + [END_TOKEN]
        confs.append(conf)
        vpathes.append(seq)
    return confs, vpathes


def find_unique_pathes(scores, pathes, output_max=False):
    path_dict = {}
    for idx, (s, p) in enumerate(zip(scores, pathes)):
        path_key = ' '.join([str(pi) for pi in p])
        if path_key in path_dict:
            path_dict[path_key].append(idx)
        else:
            path_dict[path_key] = [idx]
    # get unique path and scores
    new_pathes, new_scores = [], []
    for v in path_dict.values():
        new_pathes.append(pathes[v[0]])
        new_scores.append(max([scores[_idx] for _idx in v]))
    if output_max:
        idx = np.argmax(new_scores)
        new_scores = [new_scores[idx]]
        new_pathes = [new_pathes[idx]]
    return new_scores, new_pathes


def split_answers_from_question(pathes):
    _split_token_id = 169
    answers = []
    questions = []
    for p in pathes:
        ids = np.where(np.array(p) == _split_token_id)[0]
        if not ids:  # if not split token
            print('no split token, skipping')
            continue
        idx = ids[0]
        tmp_a = p[:idx]
        tmp_q = p[idx + 1:]
        answers.append(tmp_a)
        questions.append(tmp_q)
    return answers, questions


class TopAnswerVocab(object):
    def __init__(self):
        top_voc_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        vocab = []
        with open(top_voc_file, 'r') as fs:
            for line in fs:
                word = line.strip()
                vocab.append(word)
        self.vocab = {word: i for (i, word) in enumerate(vocab)}
        self.vocab_size = len(self.vocab)

    def match(self, answer):
        if answer in self.vocab:
            return self.vocab[answer]
        else:
            return self.vocab_size


def ivqa_decoding_beam_search(checkpoint_path=None, subset=FLAGS.subset):
    model_config = ModelConfig()
    _model_suffix = 'var_' if FLAGS.use_var else ''
    res_file = 'data4/%sv2qa_%s_questions.json' % (_model_suffix, FLAGS.subset)
    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    create_fn = create_reader('VQG-Var', phase='test')

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset='trainval',
                                    quest_vocab_file='data/vqa_trainval_merged_word_counts.txt')
    top_ans_ctx = TopAnswerVocab()

    # get data reader
    batch_size = 32
    reader = create_fn(batch_size=batch_size, subset=subset,
                       version=FLAGS.test_version)

    if checkpoint_path is None:
        if FLAGS.use_var:  # variational models
            ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
        else:  # standard models
            ckpt_dir = FLAGS.checkpoint_dir % ('kprestval', FLAGS.model_type)
        # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL/'
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        checkpoint_path = ckpt.model_checkpoint_path

    mode = 'sampling' if FLAGS.use_var else 'beam'

    # Build model
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, mode)
        model.build()
        # Restore from checkpoint
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    num_batches = reader.num_batches

    print('Running beam search inference...')
    results = []
    extend_questions = []
    ext_top_answer = []
    extended_question_ids = []
    for i in range(num_batches):
        print('iter: %d/%d' % (i, num_batches))
        outputs = reader.get_test_batch()

        # inference
        quest_ids, image_ids = outputs[-2:]
        scores, pathes = model.greedy_inference(outputs[:-2], sess)
        scores, pathes = post_process_prediction(scores, pathes,
                                                 add_start_end=False)

        # process for each sample
        _this_batch_size = quest_ids.shape[0]
        num_sampled = int(len(pathes) / _this_batch_size)
        _noise_offset = np.arange(0, num_sampled, dtype=np.int32) * _this_batch_size
        for _s_id in range(_this_batch_size):
            _index = _noise_offset + _s_id
            try:
                cur_scores = [scores[_idx] for _idx in _index]
                cur_pathes = [pathes[_idx] for _idx in _index]
            except Exception, e:
                print(str(e))
                pdb.set_trace()
            # print('Before:')
            # print(len(cur_scores))
            # pdb.set_trace()
            cur_scores, cur_pathes = find_unique_pathes(cur_scores, cur_pathes)
            cur_ans, cur_quests = split_answers_from_question(cur_pathes)
            # print('After:')
            # print(len(cur_scores))
            # pdb.set_trace()
            question_id = int(quest_ids[_s_id])
            image_id = image_ids[_s_id]

            for _pid, (ca, cq) in enumerate(zip(cur_ans, cur_quests)):
                a = to_sentence.index_to_question(ca)
                # q = to_sentence.index_to_question(cq)
                aid = top_ans_ctx.match(a)
                if aid == 2000:
                    # print('OOV')
                    continue
                extend_questions.append(cq)
                ext_top_answer.append(aid)
                extended_question_ids.append([question_id, _pid])
                # print('Q:%s A: %s' % (q, a))
            # pdb.set_trace()

            # for _pid, path in enumerate(cur_pathes):
            #     sentence = to_sentence.index_to_question(path)
            #     extended_question_ids.append([question_id, _pid])
            #     aug_quest_id = question_id * 100 + _pid
            #     res_i = {'image_id': int(image_id),
            #              'question_id': aug_quest_id,
            #              'question': sentence}
            #     results.append(res_i)
            # extend_questions += cur_pathes

    save_json(res_file, results)
    ext_quest_arr, ext_quest_len = put_to_array(extend_questions)
    ext_quest_ids = np.array(extended_question_ids, dtype=np.int32)
    ext_top_answer = np.array(ext_top_answer, dtype=np.int32)
    save_hdf5('data4/%sv2qa_%s_question_tokens.data' % (_model_suffix,
                                                        FLAGS.subset),
              {'ext_quest_arr': ext_quest_arr,
               'ext_quest_len': ext_quest_len,
               'ext_quest_ids': ext_quest_ids,
               'ext_top_answer': ext_top_answer})
    return res_file


def main(_):
    from watch_model import ModelWatcher
    subset = FLAGS.subset

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = ivqa_decoding_beam_search(checkpoint_path=model_path,
                                                 subset=subset)
            cider = evaluate_oracle(res_file)
        return cider

    ckpt_dir = FLAGS.checkpoint_dir % (FLAGS.version, FLAGS.model_type)
    # ckpt_dir = '/import/vision-ephemeral/fl302/models/v2_kpvaq_VAQ-RL_ft/'
    # res_file = ivqa_decoding_beam_search(None,
    #                                      subset=subset)
    # print(ckpt_dir)
    watcher = ModelWatcher(ckpt_dir, test_model)
    watcher.run()


if __name__ == '__main__':
    ivqa_decoding_beam_search()
    # tf.app.run()
