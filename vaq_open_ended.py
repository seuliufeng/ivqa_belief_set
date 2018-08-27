from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json

from util import get_model_iteration
from config import QuestionGeneratorConfig
from record_reader import TFRecordDataFetcher
from w2v_answer_encoder import CandidateAnswerManager
from vaq_model_generator import create_model_fn
from inference_utils.question_generator_util import SentenceGenerator

tf.flags.DEFINE_string("model_type", "VA-lstm",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/vaq_incept_lstm_ans_enc",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("input_files", "data/vqa_incept_mscoco_dev.tfrecords",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("result_file", "result/vaq_mc_incept_%d.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def restore_model(sess, checkpoint_path):
    saver = tf.train.Saver(var_list=tf.all_variables())
    saver.restore(sess, checkpoint_path)


def padding_sequence(coding):
    ans_len = np.array([len(e) for e in coding], dtype=np.int64)
    num = ans_len.size
    ans_seq = np.zeros([num, ans_len.max()], dtype=np.int64)
    for a_row, seq in zip(ans_seq, coding):
        a_row[:len(seq)] = seq
    return ans_seq, ans_len


def build_mc_data_word2vec(reader_outs, word_vec):
    num_cands = word_vec.shape[0]

    def _duplicate_vector(datum):
        return np.tile(datum.reshape([1, -1]), [num_cands, 1])

    im_ids, quest_id, im_feat, ans_w2v, quest_ids, ans_ids = reader_outs
    im_feat = _duplicate_vector(im_feat)
    quest_in = _duplicate_vector(quest_ids[:-1])
    quest_targ = _duplicate_vector(quest_ids[1:])
    quest_mask = np.array(np.ones_like(quest_targ), dtype=np.int64)
    return im_feat, word_vec, quest_in, quest_targ, quest_mask


def build_mc_data_sequence(reader_outs, sequence):
    num_cands = len(sequence)
    ans_seq, ans_len = padding_sequence(sequence)

    def _duplicate_vector(datum):
        return np.tile(datum.reshape([1, -1]), [num_cands, 1])

    im_ids, quest_id, im_feat, ans_w2v, quest_ids, ans_ids = reader_outs
    im_feat = _duplicate_vector(im_feat)
    quest_in = _duplicate_vector(quest_ids[:-1])
    quest_targ = _duplicate_vector(quest_ids[1:])
    quest_mask = np.array(np.ones_like(quest_targ), dtype=np.int64)
    return im_feat, ans_seq, ans_len, quest_in, quest_targ, quest_mask


def build_mc_reader_proc_fn(coding):
    if coding == 'word2vec':
        return build_mc_data_word2vec
    elif coding == 'sequence':
        return build_mc_data_sequence
    else:
        raise Exception('unknown coding')


def test(T=3.0, num_cands=10):
    # Build the inference graph.
    cand_file = 'result/vqa_cands.json'
    config = QuestionGeneratorConfig()
    reader = TFRecordDataFetcher(FLAGS.input_files,
                                 config.image_feature_key)

    # Create model creator
    model_creator = create_model_fn(FLAGS.model_type)

    # create multiple choice question manger
    oe_manager = CandidateAnswerManager(cand_file, max_num_cands=10)

    # Create reader post-processing function
    reader_post_proc_fn = build_mc_reader_proc_fn(model_creator.ans_coding)

    g = tf.Graph()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)
    with g.as_default():
        model = model_creator(config, phase='evaluate')
        model.build()

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)

    result = []
    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        saver = tf.train.Saver(var_list=tf.all_variables())
        saver.restore(sess, checkpoint_path)

        itr = 0
        while not reader.eof():
            outputs = reader.pop_batch()
            im_ids, quest_id, im_feat, ans_w2v, quest_ids, ans_ids = outputs
            oe_ans, oe_coding, scores = oe_manager.get_answer_sequence(quest_id)
            inputs = reader_post_proc_fn(outputs, oe_coding)
            perplexity, state = sess.run([model.likelihood, model.final_decoder_state],
                                         feed_dict=model.fill_feed_dict(inputs))
            perplexity = perplexity.reshape(inputs[-1].shape)
            loss = perplexity[:, :-1].mean(axis=1)
            weight = np.exp(-loss * T)
            weight = weight / weight.sum()  # l1 normalise
            score = scores * weight
            score = score[:num_cands]

            question = to_sentence.index_to_question(quest_ids)
            answer = to_sentence.index_to_answer(ans_ids)
            top1_ans = oe_ans[score.argmax()]
            result.append({u'answer': top1_ans, u'question_id': quest_id})

            if itr % 100 == 0:
                print('============== %d ============' % itr)
                print('image id: %d, question id: %d' % (im_ids, quest_id))
                print('question\t: %s' % question)
                print('answer\t: %s' % answer)
                top_k_ids = (-score).argsort()[:3].tolist()
                print('VQA answer\t: %s' % oe_ans[0])
                for i, idx in enumerate(top_k_ids):
                    t_mc_ans = oe_ans[idx]
                    print('VAQ answer <%d>\t: %s (%0.2f)' % (i, t_mc_ans, weight[idx]))

            itr += 1

        quest_ids = [res[u'question_id'] for res in result]
        # save results
        tf.logging.info('Saving results')
        res_file = FLAGS.result_file % get_model_iteration(checkpoint_path)
        json.dump(result, open(res_file, 'w'))
        return res_file, quest_ids


def main(_):
    res_file, quest_ids = test()
    from vqa_eval import evaluate_model
    evaluate_model(res_file, quest_ids)


if __name__ == "__main__":
    tf.app.run()
