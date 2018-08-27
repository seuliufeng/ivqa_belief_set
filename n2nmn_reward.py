import os
# from vqa_data_provider_layer import LoadVQADataProvider
# import caffe
import numpy as np
from config import VOCAB_CONFIG
import tensorflow as tf
from models_vqa.nmn3_assembler import Assembler
from models_vqa.nmn3_model import NMN3Model

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id

# Module parameters
H_feat = 14
W_feat = 14
D_feat = 2048
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 1000
num_layers = 2
T_encoder = 26
T_decoder = 13
N = 50
use_qpn = True
reduce_visfeat_dim = False

# ------------------------------ TEXT PROCESSING UTIL ------------------------------
import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def tokenize(sentence):
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds


class TopAnswerVersionConverter(object):
    def __init__(self):
        self.debug = False
        src_vocab_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        dst_vocab_file = '/usr/data/fl302/code/n2nmn/exp_vqa/data/answers_vqa.txt'
        src_vocab = self._load_top_answers(src_vocab_file)
        dst_vocab = self._load_top_answers(dst_vocab_file)
        dst_vocab2_idx = {word: i for (i, word) in enumerate(dst_vocab)}
        mapping = []
        for word in src_vocab:
            if word in dst_vocab2_idx:
                mapping.append(dst_vocab2_idx[word])
            else:
                mapping.append(2000)
        mapping.append(2000)  # add oov
        self.top_ans_src_to_dst = np.array(mapping, dtype=np.int32)
        self.src_vocab = src_vocab + ['<unk>']
        self.dst_vocab = dst_vocab

    @staticmethod
    def _load_top_answers(vocab_file):
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
            return [line.strip() for line in reverse_vocab]

    def convert(self, src_top_ans_ids):
        dst_top_ans_ids = self.top_ans_src_to_dst[src_top_ans_ids]
        if self.debug:
            num = src_top_ans_ids.size
            vis_ids = np.random.choice(num, 3, replace=False)
            for idx in vis_ids:
                print('v1: %s, v2: %s' % (self.src_vocab[src_top_ans_ids[idx]],
                                          self.dst_vocab[dst_top_ans_ids[idx]]))
        return dst_top_ans_ids


# ------------------------------      MODEL WRAPPER     ------------------------------
class N2MNModel(object):
    def __init__(self):
        self.T_encoder = 26
        data_root = '/usr/data/fl302/code/n2nmn/exp_vqa/data'
        snapshot_file = '/usr/data/fl302/code/n2nmn/exp_vqa/tfmodel/vqa_rl_gt_layout/00040000'
        self.vocab_question_file = os.path.join(data_root, 'vocabulary_vqa.txt')
        self.vocab_layout_file = os.path.join(data_root, 'vocabulary_layout.txt')
        self.vocab_answer_file = os.path.join(data_root, 'answers_vqa.txt')
        self.vocab_dict = VocabDict(self.vocab_question_file)
        self.answer_dict = VocabDict(self.vocab_answer_file)
        self.answer_word_list = self.answer_dict.word_list
        self.src_ans_id2dst = TopAnswerVersionConverter()

        self.assembler = Assembler(self.vocab_layout_file)
        num_vocab_txt = self.vocab_dict.num_vocab
        num_vocab_nmn = len(self.assembler.module_names)
        num_choices = self.answer_dict.num_vocab
        # pdb.set_trace()

        # Start the session BEFORE importing tensorflow_fold
        # to avoid taking up all GPU memory
        with tf.Graph().as_default():
            self.input_seq_batch = tf.placeholder(tf.int32, [None, None])
            self.seq_length_batch = tf.placeholder(tf.int32, [None])
            self.image_feat_batch = tf.placeholder(tf.float32, [None, H_feat, W_feat, D_feat])
            self.expr_validity_batch = tf.placeholder(tf.bool, [None])

            # build model
            self.nmn3_model_tst = NMN3Model(
                self.image_feat_batch, self.input_seq_batch,
                self.seq_length_batch, T_decoder=T_decoder,
                num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
                num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
                lstm_dim=lstm_dim, num_layers=num_layers,
                assembler=self.assembler,
                encoder_dropout=False,
                decoder_dropout=False,
                decoder_sampling=False,
                num_choices=num_choices,
                use_qpn=use_qpn, qpn_dropout=False,
                reduce_visfeat_dim=reduce_visfeat_dim)

            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=False, log_device_placement=False))

            snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
            snapshot_saver.restore(self.sess, snapshot_file)

    def _prepare_question(self, questions):
        actual_batch_size = len(questions)
        input_seq_batch = np.zeros((self.T_encoder, actual_batch_size), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)

        for n, question in enumerate(questions):
            question_tokens = tokenize(question)
            question_inds = [self.vocab_dict.word2idx(w) for w in question_tokens]
            seq_length = len(question_inds)
            input_seq_batch[:seq_length, n] = question_inds
            seq_length_batch[n] = seq_length
        return input_seq_batch, seq_length_batch

    def _prepare_images(self, image_id, questions):
        num_tiles = len(questions)
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        filename = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
        f = f.transpose((1, 2, 0))[np.newaxis, ::]
        return np.tile(f, [num_tiles, 1, 1, 1])

    def get_top_answer_ids(self, src_top_ans_ids):
        return self.src_ans_id2dst.convert(src_top_ans_ids)

    def inference(self, image_id, questions):
        nmn3_model_tst = self.nmn3_model_tst
        # image batch
        image_batch = self._prepare_images(image_id, questions)
        # question batch
        seq, seq_length = self._prepare_question(questions)
        # pdb.set_trace()
        # set up input and output tensors
        h = self.sess.partial_run_setup(
            [nmn3_model_tst.predicted_tokens, nmn3_model_tst.scores],
            [self.input_seq_batch, self.seq_length_batch, self.image_feat_batch,
             nmn3_model_tst.compiler.loom_input_tensor, self.expr_validity_batch])

        # Part 0 & 1: Run Convnet and generate module layout
        tokens = self.sess.partial_run(h, nmn3_model_tst.predicted_tokens,
                                       feed_dict={self.input_seq_batch: seq,
                                                  self.seq_length_batch: seq_length,
                                                  self.image_feat_batch: image_batch})

        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = self.assembler.assemble(tokens)
        # Build TensorFlow Fold input for NMN
        expr_feed = nmn3_model_tst.compiler.build_feed_dict(expr_list)
        expr_feed[self.expr_validity_batch] = expr_validity_array

        # Part 2: Run NMN and learning steps
        scores_val = self.sess.partial_run(h, nmn3_model_tst.scores, feed_dict=expr_feed)
        scores_val[:, 0] = -1e10  # remove <unk> answer

        # compute accuracy
        predictions = np.argmax(scores_val, axis=1)
        scores = np.max(scores_val, axis=1)
        pred_answers = [self.answer_word_list[p] for p in predictions]
        return pred_answers, scores

    def inference_batch(self, image_batch, questions):
        nmn3_model_tst = self.nmn3_model_tst
        # question batch
        seq, seq_length = self._prepare_question(questions)
        # pdb.set_trace()
        # set up input and output tensors
        h = self.sess.partial_run_setup(
            [nmn3_model_tst.predicted_tokens, nmn3_model_tst.scores],
            [self.input_seq_batch, self.seq_length_batch, self.image_feat_batch,
             nmn3_model_tst.compiler.loom_input_tensor, self.expr_validity_batch])

        # Part 0 & 1: Run Convnet and generate module layout
        tokens = self.sess.partial_run(h, nmn3_model_tst.predicted_tokens,
                                       feed_dict={self.input_seq_batch: seq,
                                                  self.seq_length_batch: seq_length,
                                                  self.image_feat_batch: image_batch})

        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = self.assembler.assemble(tokens)
        # Build TensorFlow Fold input for NMN
        expr_feed = nmn3_model_tst.compiler.build_feed_dict(expr_list)
        expr_feed[self.expr_validity_batch] = expr_validity_array

        # Part 2: Run NMN and learning steps
        scores = self.sess.partial_run(h, nmn3_model_tst.scores, feed_dict=expr_feed)
        return scores


class N2NMNReward(object):
    def __init__(self, to_sentence):
        self.to_sentence = to_sentence
        self.model = N2MNModel()

    def get_reward(self, sampled, inputs):
        images, res5c, ans, ans_len, top_ans_ids = inputs
        images_aug = []
        top_ans_ids_aug = []
        answer_aug = []
        answer_len_aug = []
        pathes = []
        top_ans_ids = self.model.get_top_answer_ids(top_ans_ids)
        res5c_aug = []
        for _idx, (ps, f) in enumerate(zip(sampled, res5c)):
            _n = len(ps)
            # f = (f / np.sqrt((f ** 2).sum()))
            # f = f[np.newaxis, ::].transpose((0, 3, 1, 2))  # convert NxHxWxC to NxCxHxW
            res5c_aug.append(np.tile(f[np.newaxis, ::], [_n, 1, 1, 1]))

            for p in ps:
                if p[-1] == END_TOKEN:
                    pathes.append(p[1:-1])  # remove start end token
                else:
                    pathes.append(p[1:])  # remove start end token
                images_aug.append(images[_idx][np.newaxis, :])
                answer_aug.append(ans[_idx][np.newaxis, :])
                answer_len_aug.append(ans_len[_idx])
                top_ans_ids_aug.append(top_ans_ids[_idx])

        questions = [self.to_sentence.index_to_question(p) for p in pathes]
        # put to arrays
        images_aug = np.concatenate(images_aug)
        res5c_aug = np.concatenate(res5c_aug)
        answer_aug = np.concatenate(answer_aug).astype(np.int32)
        top_ans_ids_aug = np.array(top_ans_ids_aug)
        answer_len_aug = np.array(answer_len_aug, dtype=np.int32)
        # run inference in VQA
        scores = self.model.inference_batch(res5c_aug, questions)
        _this_batch_size = scores.shape[0]
        vqa_scores = scores[np.arange(_this_batch_size), top_ans_ids_aug]
        is_valid = top_ans_ids_aug != 0
        return vqa_scores, [images_aug, answer_aug, answer_len_aug, is_valid]


if __name__ == '__main__':
    model = MCBModel(batch_size=2)
    scores = model.inference([96549, 96549], ['Is this a fighter plane ?',
                                              'Is this a commercial plane?'])
    import pdb

    pdb.set_trace()
