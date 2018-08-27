from __future__ import division
import tensorflow as tf
import numpy as np
import os
from util import save_json

from inference_utils import caption_generator
from inference_utils.question_generator_util import SentenceGenerator
from config import ModelConfig

from mtl_data_fetcher import AttentionTestDataFetcher as Reader
from vaq_inference_wrapper import InferenceWrapper

# TEST_SET = 'test-dev'
TEST_SET = 'dev'

tf.flags.DEFINE_string("model_type", "VAQ-van",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/vaq_%s",
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
    use_answer_sequence = 'lstm' in FLAGS.model_type or FLAGS.model_type == 'VAQ-A'
    config.model_type = FLAGS.model_type

    # build data reader
    reader = Reader(batch_size=1, subset='dev', output_attr=True, output_im=False,
                    output_qa=True, output_capt=False, output_ans_seq=use_answer_sequence)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    res_file = 'result/quest_vaq_%s.json' % FLAGS.model_type.upper()
    res_file = 'result/quest_vaq_%s.json' % FLAGS.model_type.upper()
    # build and restore model
    model = InferenceWrapper()
    restore_fn = model.build_graph_from_config(config, checkpoint_path)

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    restore_fn(sess)

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)
    generator = caption_generator.CaptionGenerator(model, to_sentence.question_vocab)

    results = []
    print('Running inference on split %s...' % TEST_SET)
    num_batches = reader.num_batches
    for i in range(num_batches):
        outputs = reader.get_test_batch()
        im_feed, quest, _, ans_feed, quest_id, image_id = outputs
        if ans_feed == 2000:
            continue
        image_id = int(image_id)
        quest_id = int(quest_id)
        im_feed = np.squeeze(im_feed)
        quest = np.squeeze(quest)
        # print('\n============== %d ============\n' % i)
        captions = generator.beam_search(sess, [im_feed, ans_feed])
        question = to_sentence.index_to_question(quest.tolist())
        answer = to_sentence.index_to_top_answer(ans_feed)
        print('============== %d ============' % i)
        print('image id: %d, question id: %d' % (image_id, quest_id))
        print('question\t: %s' % question)
        tmp = []
        for c, g in enumerate(captions[0:3]):
            quest = to_sentence.index_to_question(g.sentence)
            tmp.append(quest)
            print('<question %d>\t: %s' % (c, quest))
        print('answer\t: %s\n' % answer)

        caption = captions[0]
        sentence = to_sentence.index_to_question(caption.sentence)
        res_i = {'image_id': image_id, 'question_id': quest_id, 'question': sentence}
        results.append(res_i)
    save_json(res_file, results)
    return res_file


def main(_):

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = test(model_path)
        return evaluate_question(res_file, subset='dev')

    test_model(None)


if __name__ == '__main__':
    tf.app.run()
