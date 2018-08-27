from nltk.tokenize import word_tokenize
from inference_utils import vocabulary


def _tokenize_sentence(sentence):
    return word_tokenize(str(sentence).lower())


def serialize_path(path):
    return ' '.join([str(t) for t in path])


class AnswerTokenToTopAnswer(object):
    def __init__(self):
        self._load_answer_vocab()
        self._load_top_answer_vocab()

    def _load_answer_vocab(self):
        ans_voc_file = 'data/vqa_trainval_answer_word_counts.txt'
        self.ans_voc = vocabulary.Vocabulary(ans_voc_file)

    def _load_top_answer_vocab(self):
        top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        answer_vocab = []
        with open(top_ans_file, 'r') as fs:
            for line in fs:
                answer_vocab.append(line.strip())
        self.answer_vocab = {w: i for i, w in enumerate(answer_vocab)}
        top_answer_voc = {}
        for i, _answer in enumerate(answer_vocab):
            tokens = _tokenize_sentence(_answer)
            token_ids = [self.ans_voc.word_to_id(word) for word in tokens]
            key = serialize_path(token_ids)
            top_answer_voc[key] = i
        self.top_answer_voc = top_answer_voc

    def direct_query(self, query):
        if query in self.answer_vocab:
            return self.answer_vocab[query]
        else:
            return 2000

    def get_top_answer(self, ans):
        inds = []
        for a in ans:
            query = serialize_path(a)
            if query in self.top_answer_voc:
                inds.append(self.top_answer_voc[query])
            else:
                inds.append(2000)
        return inds
