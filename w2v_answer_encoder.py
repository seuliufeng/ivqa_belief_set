from word2vec_util import Word2VecEncoder
import tensorflow as tf
import numpy as np
import json
import os
try:
    from nltk.tokenize import word_tokenize
except:
    pass
from inference_utils import vocabulary
from config import QuestionGeneratorConfig
import scipy.spatial.distance as ssd

_CONFIG = QuestionGeneratorConfig()
_IM_ROOT = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco/'


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


class AnswerEncoder(object):
    def __init__(self, top_answer_file='data/vqa_trainval_top2000_answers.txt'):
        print('Loading answer encoder')
        # model_file = 'data/vqa_word2vec_model.pkl'
        model_file = '/import/vision-ephemeral/fl302/code/VQA-tensorflow/data/vqa_word2vec_model.pkl'
        # self._encoder = Word2VecEncoder(model_file)
        self._encoder = None
        self._top_answer_file = top_answer_file
        self._top_ans_vocab = None
        self._top_ans_w2v = None
        self._ans_word_vocab = None
        self._top_ans_seq = None
        self._max_ans_vocab_size = _CONFIG.ans_vocab_size
        self._max_quest_vocab_size = _CONFIG.vocab_size
        self._load_top_answers(self._top_answer_file)
        self._load_answer_vocabulary()

    def set_answer_vocab(self, vocab_file):
        self._load_answer_vocabulary(vocab_file)

    def _load_top_answers(self, vocab_file):
        print('Loading top answer vocabulary...')
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
            self._top_ans_vocab = [line.strip() for line in reverse_vocab]
        self._top_ans_vocab.append('UNK')
        self._top_answer_to_index = {ans: idx for idx, ans in enumerate(self._top_ans_vocab)}

    def _load_answer_vocabulary(self, vocab_file=None):
        print('Loading answer words vocabulary...')
        if vocab_file is None:
            vocab_file = 'data/vqa_%s_answer_word_counts.txt' % 'trainval'
        print(vocab_file)
        self._ans_word_vocab = vocabulary.Vocabulary(vocab_file)

    def _create_top_ans_sequence_pool(self):
        self._top_ans_seq = self.encode_to_sequence(self._top_ans_vocab)

    def _create_top_ans_word_vector(self):
        index = []
        for ans in self._top_ans_vocab:
            w2v = self._encoder.encode(ans)
            index.append(w2v.reshape([1, -1]))
        self._top_ans_w2v = np.concatenate(index)

    def encode_by_index(self, index):
        if self._top_ans_w2v is None:
            self._create_top_ans_word_vector()
        return self._top_ans_w2v[index]

    def encode_by_answer(self, answers):
        output = []
        for ans in answers:
            if type(ans) != list:
                ans = _tokenize_sentence(ans)
            output.append(self._encoder.encode(ans).reshape([1, -1]))
        return np.concatenate(output)

    def encode_to_sequence(self, ans):
        ans_seq = []
        for sentence in ans:
            tokenized = _tokenize_sentence(sentence) if type(sentence) != list else sentence
            ids = [self._ans_word_vocab.word_to_id(word) for word in tokenized]
            ans_seq.append([min(self._max_ans_vocab_size, i) for i in ids])
        return ans_seq

    def get_top_answers(self, index):
        return [self._top_ans_vocab[i] for i in index]

    def get_nearest_top_answer_index(self, ans):
        if ans in self._top_answer_to_index:
            return self._top_answer_to_index[ans]
        # if not in top answers
        if self._top_ans_w2v is None:
            self._create_top_ans_word_vector()
        f = self._encoder.encode(ans)
        d2 = ssd.cdist(f[np.newaxis, :], self._top_ans_w2v)
        return d2.flatten().argmin()

    def encode_to_sequence_by_index(self, index):
        if self._top_ans_seq is None:
            self._create_top_ans_sequence_pool()
        return [self._top_ans_seq[i] for i in index]


class MultiChoiceQuestionManger(object):
    def __init__(self, subset='val', load_ans=True, answer_coding='word2vec',
                 top_ans_file='data/vqa_trainval_top2000_answers.txt'):
        self._subset = subset
        self._load_ans = load_ans
        self._ans_coding = answer_coding
        self._mc_vocab = {}
        self._gt_ans_vocab = {}
        self._quest_id2image_id = {}
        self._image_id2quest_ids = {}
        self._quest_id2im_file = {}
        self._quest_vocab = {}
        self._quest_id2ans_type = {}
        self._answer_type2id = {}
        self._quest_ids = []
        if 'train' in subset:
            self._load_question_file('train')
            if self._load_ans:
                self._load_annotation('train')
        if 'val' in subset:
            self._load_question_file('val')
            if self._load_ans:
                self._load_annotation('val')
        # image id 2 question ids
        for quest_id, image_id in self._quest_id2image_id.iteritems():
            self._image_id2quest_ids.setdefault(image_id, []).append(quest_id)
        if self._load_ans:
            self._encoder = AnswerEncoder(top_answer_file=top_ans_file)
        self._answer_type2id = {k: i for i, k in enumerate(self._answer_type2id.keys())}
        self._id2answer_type = {v: k for k, v in self._answer_type2id.iteritems()}

    def _load_annotation(self, subset):
        print('Loading annotation files [%s]...' % subset)
        ann_file = 'data/annotations/mscoco_%s2014_annotations.json' % subset
        d = json.load(open(ann_file, 'r'))
        for info in d['annotations']:
            quest_id = info['question_id']
            im_id = info['image_id']
            ans = info['multiple_choice_answer']
            self._quest_id2ans_type[quest_id] = info['answer_type']
            self._answer_type2id.setdefault(info['answer_type'], [])
            im_file = '%s2014/COCO_%s2014_%012d.jpg' % (subset, subset, im_id)
            self._gt_ans_vocab[quest_id] = ans
            self._quest_id2im_file[quest_id] = os.path.join(_IM_ROOT, im_file)
            self._quest_id2image_id[quest_id] = im_id

    def _load_question_file(self, subset):
        print('Loading question files [%s]...' % subset)
        quest_file = 'data/annotations/MultipleChoice_mscoco_%s2014_questions.json' % subset
        d = json.load(open(quest_file, 'r'))
        questions = d['questions']
        print('Tokenize candidate answers...')
        for info in questions:
            question_id = info['question_id']
            choices = info['multiple_choices']
            quest = info['question']
            self._mc_vocab[question_id] = choices
            self._quest_vocab[question_id] = quest
        self._quest_ids = self._quest_vocab.keys()

    def image_id2quest_ids(self, image_id):
        return self._image_id2quest_ids[image_id]

    def id_to_answer_type(self, id):
        return self._id2answer_type[id]

    def get_image_file(self, quest_id):
        return self._quest_id2im_file[quest_id]

    def get_image_id(self, quest_id):
        return self._quest_id2image_id[quest_id]

    def get_answer_type(self, quest_id):
        return self._quest_id2ans_type[quest_id]

    def get_answer_type_coding(self, quest_id):
        ans_type = self.get_answer_type(quest_id)
        return self._answer_type2id[ans_type]

    def get_question_ids(self):
        return self._quest_ids

    def get_question(self, quest_id):
        return self._quest_vocab[quest_id]

    def get_gt_answer(self, quest_id):
        return self._gt_ans_vocab[quest_id]

    def get_gt_answer_and_sequence_coding(self, quest_id):
        ans = self.get_gt_answer(quest_id)
        seq = self._encoder.encode_to_sequence([ans])
        return ans, seq

    def get_gt_answer_and_word2vec(self, quest_id):
        ans = self.get_gt_answer(quest_id)
        w2v = self._coding_to_word_vectors([ans])
        return ans, w2v

    def get_binary_label(self, quest_id):
        cands = self.get_candidate_answers(quest_id)
        ans = self.get_gt_answer(quest_id)
        return np.array([a == ans for a in cands])

    def get_candidate_answers(self, quest_id):
        return self._mc_vocab[quest_id]

    def get_candidate_answer_and_word_coding(self, quest_id):
        ans = self.get_candidate_answers(quest_id)
        if self._ans_coding == 'word2vec':
            return ans, self._coding_to_word_vectors(ans)
        elif self._ans_coding == 'sequence':
            return ans, self._coding_to_sequences(ans)
        else:
            raise Exception('unknown word coding')

    def _coding_to_word_vectors(self, ans):
        return self._encoder.encode_by_answer(ans)

    def _coding_to_sequences(self, ans):
        return self._encoder.encode_to_sequence(ans)

    @property
    def encoder(self):
        return self._encoder


def get_top_answer_w2v():
    anc_enc = AnswerEncoder()
    anc_enc.encode_by_index([0])  # dummy
    init_w = anc_enc._top_ans_w2v
    return init_w


class CandidateAnswerManager(object):
    def __init__(self, res_file, max_num_cands=10, coding='sequence'):
        assert (coding in ['word2vec', 'sequence'])
        self._top_ans_vocab = None
        self._q2ans = None
        self._q2scores = None
        self._max_n_cands = max_num_cands
        self._ans_encoder = AnswerEncoder()
        self._load_vqa_candidates(res_file)
        self._load_vqa_top_answers()

    def get_answer_coding_word2vec(self, quest_id):
        quest_id = str(quest_id)
        index = self._q2ans[quest_id][:self._max_n_cands]
        ans = self._ans_encoder.get_top_answers(index)
        return ans, self._ans_encoder.encode_by_index(index), \
               self._q2scores[quest_id][:self._max_n_cands]

    def get_answer_sequence(self, quest_id):
        quest_id = str(quest_id)
        index = self._q2ans[quest_id][:self._max_n_cands]
        ans = self._ans_encoder.get_top_answers(index)
        return ans, self._ans_encoder.encode_to_sequence_by_index(index), \
               self._q2scores[quest_id][:self._max_n_cands]

    def _load_vqa_candidates(self, res_file):
        d = json.load(open(res_file))
        self._q2ans = {k: d[k][0] for k in d.keys()}
        self._q2scores = {k: d[k][1] for k in d.keys()}

    def _load_vqa_top_answers(self):
        print('Loading top answer vocabulary...')
        vocab_file = 'data/vqa_%s_top2000_answers.txt' % 'trainval'
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
            self._top_ans_vocab = [line.strip() for line in reverse_vocab]


if __name__ == '__main__':
    ans_ctx = AnswerEncoder()
    ans_vocab = ans_ctx._top_ans_vocab
    idx = ans_ctx.get_nearest_top_answer_index('cat')
    print(ans_vocab[idx])
    idx = ans_ctx.get_nearest_top_answer_index('cats')
    print(ans_vocab[idx])
    exit(0)
    mg = MultiChoiceQuestionManger(subset='val', answer_coding='sequence')
    mg.encoder.set_answer_vocab('data/vqa_trainval_question_answer_word_counts.txt')
    quest_ids = mg.get_question_ids()
    import pdb
    pdb.set_trace()
    answer_vocab = {}
    quest_cand_index_vocab = {}
    cand_ans_sequences = []
    idx = 0
    for quest_id in quest_ids:
        cand_inds = []
        ans_cands = mg.get_candidate_answers(quest_id)
        gt_ans = mg.get_gt_answer(quest_id)
        assert(gt_ans in ans_cands)
        for ans in ans_cands:
            if ans == gt_ans:  # we don't want correct answer in it
                continue
            if ans in answer_vocab:
                ans_idx = answer_vocab[ans]
            else:
                answer_vocab[ans] = idx
                code = mg.encoder.encode_to_sequence([ans])[0]
                cand_ans_sequences.append(code)
                ans_idx = idx
                idx += 1
            cand_inds.append(ans_idx)
        quest_cand_index_vocab[quest_id] = cand_inds
    from build_vqa_retrieval_data import _put_sequence_to_matrix
    cand_arr, cand_len = _put_sequence_to_matrix(cand_ans_sequences)
    from util import save_hdf5, save_json

    meta_file = 'data/vqa_retrieval_cst_ans_mscoco_train.meta'
    data_file = 'data/vqa_retrieval_cst_ans_mscoco_train.data'
    save_json(meta_file, {'quest_id2cand_index': quest_cand_index_vocab})
    save_hdf5(data_file, {'cand_arr': cand_arr, 'cand_len': cand_len})

    # put words to a dictionary
    print('collected %d indepent answers' % len(answer_vocab))



