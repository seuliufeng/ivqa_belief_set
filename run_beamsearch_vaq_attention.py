from __future__ import division
import tensorflow as tf
import numpy as np
import os
from util import save_json

from inference_utils import caption_generator
from inference_utils.question_generator_util import SentenceGenerator
from config import ModelConfig

from mtl_data_fetcher import AttentionTestDataFetcher as Reader
from vaq_attention_inference_wrapper import InferenceWrapper

# TEST_SET = 'test-dev'
TEST_SET = 'dev'

tf.flags.DEFINE_string("model_type", "VAQ-lstm-dec",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/kpvaq_%s",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_CONF = 0.0
END_TOKEN = 2


def token_to_sentence(to_sentence, inds):
    if inds.ndim == 1:
        inds = inds[np.newaxis, :]
    captions = []
    end_pos = (inds == END_TOKEN).argmax(axis=1)

    for i_s, e in zip(inds, end_pos):
        t_ids = i_s[:e].tolist()
        if len(t_ids) == 0:
            t_ids.append(END_TOKEN)
        s = to_sentence.index_to_question(t_ids)
        captions.append(s)
    return captions


def evaluate_question(result_file, subset='dev'):
    from eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()
    assert(subset in ['train', 'dev', 'val'])
    subset = 'train' if subset == 'train' else 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()


def test(checkpoint_path=None):
    config = ModelConfig()
    config.phase = 'other'
    config.model_type = FLAGS.model_type
    config.cell_option = 5
    # config.cell_option = 4
    beam_size = 3
    subset = 'kptest'
    # build data reader
    reader = Reader(batch_size=1, subset=subset, output_attr=True, output_im=True,
                    output_qa=True, output_capt=False, output_ans_seq=True)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    res_file = 'result/beamsearch_vaq_%s_%s.json' % (FLAGS.model_type.upper(), subset)
    # rerank_file = 'result/beamsearch_vaq_reank_cands_%s_val.json' % FLAGS.model_type.upper()
    # build and restore model
    model = InferenceWrapper()
    restore_fn = model.build_graph_from_config(config, checkpoint_path)

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    restore_fn(sess)

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)
    generator = caption_generator.CaptionGenerator(model, to_sentence.question_vocab,
                                                   beam_size=beam_size)

    results = []
    re_rank_cands = []
    print('Running inference on split %s...' % TEST_SET)
    num_batches = reader.num_batches
    for i in range(num_batches):
        outputs = reader.get_test_batch()
        im_feed, attr, quest, quest_len, _, ans_seq, ans_seq_len, quest_id, image_id = outputs

        image_id = int(image_id)
        quest_id = int(quest_id)
        im_feed = np.squeeze(im_feed)
        quest = np.squeeze(quest)
        # print('\n============== %d ============\n' % i)
        captions = generator.beam_search(sess, [im_feed, attr, ans_seq, ans_seq_len])
        question = to_sentence.index_to_question(quest.tolist())
        # answer = to_sentence.index_to_top_answer(ans_feed)
        print('============== %d ============' % i)
        print('image id: %d, question id: %d' % (image_id, quest_id))
        print('question\t: %s' % question)
        tmp, tmp_scores = [], []
        vaq_cands = {'question_id': quest_id}
        for c, g in enumerate(captions):
            quest = to_sentence.index_to_question(g.sentence)
            tmp.append(quest)
            tmp_scores.append(g.logprob)
            print('<question %d>\t: %s' % (c, quest))
        # print('answer\t: %s\n' % answer)
        vaq_cands['questions'] = tmp
        vaq_cands['confidence'] = tmp_scores
        re_rank_cands.append(vaq_cands)

        caption = captions[0]
        sentence = to_sentence.index_to_question(caption.sentence)
        res_i = {'image_id': image_id, 'question_id': quest_id, 'question': sentence}
        results.append(res_i)
    save_json(res_file, results)
    # save_json(rerank_file, re_rank_cands)
    return res_file


def main(_):
    # from run_question_generator import evaluate_question
    # from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = test(model_path)
        return evaluate_question(res_file, subset='dev')

    test_model(None)

    # ckpt_dir = FLAGS.checkpoint_dir % FLAGS.model_type
    # print(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    tf.app.run()
