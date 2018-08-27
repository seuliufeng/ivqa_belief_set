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
from vaq_sequence_inference_wrapper import InferenceWrapper as SequenceInferenceWrapper
from w2v_answer_encoder import MultiChoiceQuestionManger


# models ['VAQ-IAS', 'VAQ-lstm', 'VQG', 'VAQ-A']
tf.flags.DEFINE_string("model_type", "VQG",
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


def load_model_inferencer():
    model_type = FLAGS.model_type
    if model_type in ['VAQ-A', 'VAQ-lstm']:
        return SequenceInferenceWrapper()
    else:
        return InferenceWrapper()


def add_answer_type(quest_id, mc_ctx):
    answer_type_id = mc_ctx.get_answer_type_coding(quest_id)
    answer_type_id = np.array(answer_type_id, dtype=np.int32).reshape([1, ])
    return answer_type_id


def pre_process_inputs(inputs, mc_ctx, use_answer_type=False):
    # process input data
    im_feed, quest, quest_len, _, answer_seq, ans_seq_len, quest_id, image_id = inputs
    im_feed = np.squeeze(im_feed)
    answer_seq = np.squeeze(answer_seq)
    ans_seq_len = np.squeeze(ans_seq_len)
    quest_id, image_id = int(quest_id), int(image_id)
    quest = quest.flatten()
    # for slim
    if use_answer_type:
        answer_type = add_answer_type(quest_id, mc_ctx)
        return [im_feed, answer_type], (quest_id, image_id), quest.tolist()
    else:
        return [im_feed, answer_seq, ans_seq_len], (quest_id, image_id), quest.tolist()


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
    subset = 'kptest'
    config = ModelConfig()
    config.phase = 'other'
    use_answer_type = FLAGS.model_type in ['VAQ-IAS', 'VQG']
    config.model_type = FLAGS.model_type
    mc_ctx = MultiChoiceQuestionManger(subset='val')

    # build data reader
    reader = Reader(batch_size=1, subset=subset, output_attr=True, output_im=False,
                    output_qa=True, output_capt=False, output_ans_seq=True,
                    attr_type='res152')
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    res_file = 'result/quest_vaq_%s_%s.json' % (FLAGS.model_type.upper(), subset)
    print(res_file)
    # build and restore model
    model = load_model_inferencer()
    restore_fn = model.build_graph_from_config(config, checkpoint_path)

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    restore_fn(sess)

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)
    generator = caption_generator.CaptionGenerator(model, to_sentence.question_vocab)

    results = []
    print('Running inference on split %s...' % subset)
    num_batches = reader.num_batches
    for i in range(num_batches):
        inputs, info, quest_gt_vis = pre_process_inputs(reader.get_test_batch(), mc_ctx, use_answer_type)
        quest_id, image_id = info
        captions = generator.beam_search(sess, inputs)
        question = to_sentence.index_to_question(quest_gt_vis)
        # answer = to_sentence.index_to_top_answer(ans_feed)
        print('============== %d ============' % i)
        print('image id: %d, question id: %d' % (image_id, quest_id))
        print('question\t: %s' % question)
        tmp = []
        for c, g in enumerate(captions[0:3]):
            quest = to_sentence.index_to_question(g.sentence)
            tmp.append(quest)
            print('<question %d>\t: %s' % (c, quest))
        # print('answer\t: %s\n' % answer)

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
