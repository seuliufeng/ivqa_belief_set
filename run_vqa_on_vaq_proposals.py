from __future__ import division

from vqa_model_creater import get_model_creation_fn
from config import ModelConfig
from w2v_answer_encoder import MultiChoiceQuestionManger
from util import *
from nltk.tokenize import word_tokenize
from inference_utils import vocabulary
from inference_utils.question_generator_util import SentenceGenerator

TEST_SET = 'kpval'

tf.flags.DEFINE_string("model_type", "VQA-ST",
                       "Select a model to train.")
tf.flags.DEFINE_string("feat_type", "Res5c",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/curr_kp_%s_%s",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("result_file", "result/vqa_OpenEnded_mscoco_%s2015_baseline_results.json" % TEST_SET,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_question(result_file, subset='dev'):
    from eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()
    # assert(subset in ['train', 'dev', 'val'])
    subset = 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()


def load_question_proposals():
    cand_file = 'result/beamsearch_vaq_reank_cands_VAQ-LSTM-DEC-SUP_val.json'
    d = load_json(cand_file)
    return d


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    @property
    def unk_id(self):
        return self._unk_id


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


class SentenceEncoder(object):
    def __init__(self, type='question'):
        assert (type in ['answer', 'question'])
        self._vocab = None
        vocab_file = 'data/vqa_%s_%s_word_counts.txt' % ('trainval', type)
        self._load_vocabulary(vocab_file)

    def _load_vocabulary(self, vocab_file=None):
        print('Loading answer words vocabulary...')
        print(vocab_file)
        self._vocab = vocabulary.Vocabulary(vocab_file)

    def encode_sentences(self, sentences):
        exist_multiple = type(sentences) is list
        if exist_multiple:
            tokens_list = [self._encode_sentence(s) for s in sentences]
            return self._put_to_numpy_array(tokens_list)
        else:
            tokens = np.array(self._encode_sentence(sentences),
                              dtype=np.int32)
            token_len = np.array(tokens.size, dtype=np.int32)
            return tokens.flatten(), token_len.flatten()

    def _encode_sentence(self, sentence):
        tokens = _tokenize_sentence(sentence)
        return [self._vocab.word_to_id(word) for word in tokens]

    @staticmethod
    def _put_to_numpy_array(tokens):
        seq_len = [len(q) for q in tokens]
        max_len = max(seq_len)
        num = len(tokens)
        token_arr = np.zeros([num, max_len], dtype=np.int32)
        for i, x in enumerate(token_arr):
            x[:seq_len[i]] = tokens[i]
        seq_len = np.array(seq_len, dtype=np.int32)
        return token_arr, seq_len


def _load_answer_vocab(vocab_file='data/vqa_trainval_top2000_answers.txt'):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    reverse_vocab = [line.strip() for line in reverse_vocab]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    return Vocabulary(vocab_dict, unk_id)


class ProposalManager(object):
    def __init__(self, top_k=10):
        self.proposals = load_question_proposals()
        self.mc_ctx = MultiChoiceQuestionManger(subset='val')
        self.quest_enc = SentenceEncoder(type='question')
        self.top_answer_vocab = _load_answer_vocab()
        self.num = len(self.proposals)
        self.top_k = top_k
        self.idx = 0

    def _get_image_feature(self, image_id):
        self._im_feat_root = os.path.join(get_res5c_feature_root(),
                                          'val2014')
        filename = 'COCO_val2014_%012d.jpg' % image_id
        f = np.load(os.path.join(self._im_feat_root, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]

    def get_sample(self):
        qprop = self.proposals[self.idx]
        questions = qprop['questions'][:self.top_k]
        logprob = qprop['confidence'][:self.top_k]
        quest_id = qprop['question_id']
        # encode questions
        quest, quest_len = self.quest_enc.encode_sentences(questions)
        n_cand = len(questions)

        # encode answer
        answer = self.mc_ctx.get_gt_answer(quest_id)
        answer_idx = self.top_answer_vocab.word_to_id(answer)

        # load image
        image_id = self.mc_ctx.get_image_id(quest_id)
        im = self._get_image_feature(image_id)
        im = np.tile(im, [n_cand, 1, 1, 1])

        self.idx += 1
        return [im, quest, quest_len, answer_idx, quest_id, image_id], logprob, questions


def test(checkpoint_path=None):
    T = 1 / 200.
    EPS = 1e-12
    config = ModelConfig()
    # Get model function
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # create context and load data
    prop_cand = ProposalManager(top_k=3)
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)

    if checkpoint_path is None:
        print(FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.model_type,
                                                                     FLAGS.feat_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    model.build()
    prob = model.prob
    # sess = tf.Session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(graph=tf.get_default_graph(),
                      config=tf.ConfigProto(gpu_options=gpu_options))
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    res_file = 'result/question_re_rank_%s_%s.json' % (FLAGS.model_type.upper(), 'kpval')
    print(res_file)

    print('Running inference on split %s...' % TEST_SET)

    results = []
    num_samples = prop_cand.num
    for i in range(num_samples):
        outputs, logprob, props = prop_cand.get_sample()

        generated_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-3]))
        answer_idx = outputs[3]
        scores = generated_ans[:, answer_idx]
        # normalise
        scores = np.exp(T * scores)
        scores /= (EPS + scores.sum())

        if answer_idx == 2000:
            question = props[0]
            ans = 'UNK'
        else:
            # scores = np.array(logprob, dtype=np.float32) + scores * 0.0001
            scores = np.array(logprob, dtype=np.float32) * scores
            idx = scores.argmax()
            question = props[idx]
            ans = to_sentence.index_to_top_answer(answer_idx)

        # print question
        image_id, quest_id = outputs[-1], outputs[-2]
        print('============== %d ============' % i)
        # print answer_idx
        print('image id: %d, question id: %d' % (image_id, quest_id))
        print('question\t: %s' % question)
        vis_len = outputs[2][0]
        vis = outputs[1][0, :vis_len].flatten().tolist()
        sent = to_sentence.index_to_question(vis)

        # print('Q: %s' % sent)
        # print('A: %s' % ans)

        image_id = int(image_id)
        quest_id = int(quest_id)
        res_i = {'image_id': image_id, 'question_id': quest_id, 'question': question}
        results.append(res_i)

    save_json(res_file, results)
    return res_file


def main(_):

    def test_model(model_path):
        with tf.Graph().as_default():
            res_file = test(model_path)
        return evaluate_question(res_file, subset='kpval')

    test_model(None)


if __name__ == '__main__':
    # evaluate_question('/import/vision-ephemeral/fl302/code/VQA-tensorflow/result/val_sup_att/beamsearch_vaq_VAQ-LSTM-DEC-SUP.json', subset='kpval')
    tf.app.run()
