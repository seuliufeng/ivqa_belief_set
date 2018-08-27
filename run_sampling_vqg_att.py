from __future__ import division
import tensorflow as tf
import numpy as np
import os
from util import save_json

from inference_utils import caption_generator
from inference_utils.question_generator_util import SentenceGenerator
from config import ModelConfig

from readers.ivqa_reader_creater import create_reader
from vaq_attention_inference_wrapper import InferenceWrapper
import pdb

# TEST_SET = 'test-dev'
TEST_SET = 'dev'

tf.flags.DEFINE_string("model_type", "VQG-Att",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/v1_kpvaq2_%s",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_CONF = 0.0
END_TOKEN = 2


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


def test(checkpoint_path=None):
    config = ModelConfig()
    config.phase = 'other'
    config.model_type = FLAGS.model_type

    beam_size = 10
    subset = 'kptest'
    # build data reader
    create_fn = create_reader(FLAGS.model_type, phase='test')
    reader = create_fn(batch_size=1, subset=subset, version='v1')
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    res_file = 'result/beamsearch_%s_%s.json' % (FLAGS.model_type.upper(), subset)
    cand_file = 'result/sampling_%s_%s.json' % (FLAGS.model_type.upper(), subset)

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
    candidates = []
    print('Running inference on split %s...' % TEST_SET)
    num_batches = reader.num_batches
    for i in range(num_batches):
        outputs = reader.get_test_batch()
        im_feed, attr, ans_seq, ans_seq_len, quest_id, image_id = outputs

        image_id = int(image_id)
        quest_id = int(quest_id)
        im_feed = np.squeeze(im_feed)
        captions = generator.beam_search(sess, [im_feed, attr, ans_seq, ans_seq_len])

        print('============== %d ============' % i)
        print('image id: %d, question id: %d' % (image_id, quest_id))
        # print('question\t: %s' % question)
        tmp = []
        vaq_cands_i = {'question_id': quest_id, 'image_id': image_id}
        for c, g in enumerate(captions):
            quest = to_sentence.index_to_question(g.sentence)
            tmp.append(quest)
            print('[%02d]: %s' % (c, quest))

        vaq_cands_i['candidates'] = tmp
        candidates.append(vaq_cands_i)

        caption = captions[0]
        sentence = to_sentence.index_to_question(caption.sentence)
        res_i = {'image_id': image_id, 'question_id': quest_id, 'question': sentence}
        results.append(res_i)
    save_json(res_file, results)
    save_json(cand_file, candidates)
    return res_file


def main(_):
    # from run_question_generator import evaluate_question
    # from watch_model import ModelWatcher

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = test(model_path)
        return evaluate_question(res_file, subset='kptest', version='v1')

    test_model(None)
    pdb.set_trace()

    # ckpt_dir = FLAGS.checkpoint_dir % FLAGS.model_type
    # print(ckpt_dir)
    # watcher = ModelWatcher(ckpt_dir, test_model)
    # watcher.run()


if __name__ == '__main__':
    tf.app.run()
