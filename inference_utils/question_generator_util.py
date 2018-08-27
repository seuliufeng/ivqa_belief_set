from inference_utils import vocabulary
import tensorflow as tf


def _index_to_sentence(vocab, index):
    START_TOKEN = '<S>'
    tokens = [vocab.id_to_word(idx) for idx in index]
    if tokens[0] == START_TOKEN:
        tokens = tokens[1:-1]
    return ' '.join(tokens)


class SentenceGenerator(object):
    def __init__(self, trainset='train', ans_vocab_file=None, quest_vocab_file=None,
                 top_ans_file=None):
        if ans_vocab_file is None:
            ans_vocab_file = 'data/vqa_%s_answer_word_counts.txt' % trainset
        if quest_vocab_file is None:
            quest_vocab_file = 'data/vqa_%s_question_word_counts.txt' % trainset
        self._answer_vocab = vocabulary.Vocabulary(ans_vocab_file)
        self._quest_vocab = vocabulary.Vocabulary(quest_vocab_file)
        if top_ans_file is None:
            top_ans_file = 'data/vqa_%s_top2000_answers.txt' % trainset
        self._load_top_answers(top_ans_file)

    def _load_top_answers(self, vocab_file):
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
            self._top_ans_vocab = [line.strip() for line in reverse_vocab]

    def index_to_answer(self, sequence):
        return _index_to_sentence(self._answer_vocab, sequence)

    def index_to_question(self, sequence):
        return _index_to_sentence(self._quest_vocab, sequence)

    def index_to_top_answer(self, id):
        return self._top_ans_vocab[id]

    @property
    def question_vocab(self):
        return self._quest_vocab


def _extract_gt(capt, capt_len):
    gt = []
    for c, c_len in zip(capt, capt_len):
        tmp = c[:c_len].tolist()
        gt.append(tmp)
    return gt


class DataPreviewer(object):
    def __init__(self):
        top_ans_file = '/import/vision-ephemeral/fl302/code/' \
                       'VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        self.to_sentence = SentenceGenerator(trainset='trainval',
                                             top_ans_file=top_ans_file)

    def display(self, quest, quest_len, ans, ans_len):
        gt_quests = _extract_gt(quest, quest_len)
        gt_answers = _extract_gt(ans, ans_len)
        for i, (q, a) in enumerate(zip(gt_quests, gt_answers)):
            print('Q: %s' % self.to_sentence.index_to_question(q))
            print('A: %s' % self.to_sentence.index_to_answer(a))
        print('\n')
