import tensorflow as tf
import numpy as np
import os
import cmd
from inference_utils.question_generator_util import SentenceGenerator
from nltk.tokenize import word_tokenize
from inference_utils import vocabulary


# from mcb_wrapper import MCBModel


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

    def encode_sentence(self, sentence):
        tokens = np.array(self._encode_sentence(sentence),
                          dtype=np.int32)
        token_len = np.array(tokens.size, dtype=np.int32)
        return tokens.reshape([1, -1]), token_len.flatten()

    def _encode_sentence(self, sentence):
        tokens = _tokenize_sentence(sentence)
        return [self._vocab.word_to_id(word) for word in tokens]


class BaseVQAModel(object):
    def __init__(self, ckpt_file=None):
        top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
        self.to_sentence = SentenceGenerator(trainset='trainval',
                                             top_ans_file=top_ans_file)
        self.sent_encoder = SentenceEncoder()
        self.model = None
        self.sess = None
        self.name = ''
        self.top_k = 2

        self.answer_to_top_ans_id = None

    def _load_image(self, image_id):
        return None

    def _process_question(self, question):
        arr, arr_len = self.sent_encoder.encode_sentence(question)
        return arr, arr_len

    def inference(self, image_id, question):
        image = self._load_image(image_id)
        arr, arr_len = self._process_question(question)
        scores = self.model.inference(self.sess, [image, arr, arr_len])
        self.show_prediction(scores)
        return scores

    def get_score(self, image_id, question):
        image = self._load_image(image_id)
        arr, arr_len = self._process_question(question)
        scores = self.model.inference(self.sess, [image, arr, arr_len])
        # self.show_prediction(scores)
        scores[:, -1] = -100
        id = scores.flatten().argmax()
        # print(id)
        sc = scores.flatten().max()
        answer = self.to_sentence.index_to_top_answer(id)
        return answer, sc

    def query_score(self, image_id, question, answer):
        if self.answer_to_top_ans_id is None:
            print('Creating vocabulary')
            top_ans = self.to_sentence._top_ans_vocab
            self.answer_to_top_ans_id = {ans: idx for idx, ans in enumerate(top_ans)}

        image = self._load_image(image_id)
        arr, arr_len = self._process_question(question)
        scores = self.model.inference(self.sess, [image, arr, arr_len])
        scores = scores.flatten()
        if answer in self.answer_to_top_ans_id:
            idx = self.answer_to_top_ans_id[answer]
        else:
            print('Warning: OOV')
            idx = -1
        return float(scores[idx])

    def show_prediction(self, scores):
        scores = scores.flatten()
        inds = (-scores).argsort()[:self.top_k]
        print('%s' % self.name)
        for id in inds:
            sc = scores[id]
            answer = self.to_sentence.index_to_top_answer(id)
            print('%s: %0.2f' % (answer, sc))


class AttentionModel(BaseVQAModel):
    def __init__(self, ckpt_file='model/v1_vqa_VQA/v1_vqa_VQA_best2/model.ckpt-135000'):
        BaseVQAModel.__init__(self)
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        from models.vqa_soft_attention import AttentionModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.name = ' ------- MLB-attention ------- '

        with self.g.as_default():
            self.sess = tf.Session()
            self.model = AttentionModel(config, phase='test_broadcast')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

    def _load_image(self, image_id):
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        filename = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]


class VanillaModel(BaseVQAModel):
    def __init__(self, ckpt_file='model/kprestval_VQA-BaseNorm/model.ckpt-26000'):
        BaseVQAModel.__init__(self)
        self.top_k = 2
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        from models.vqa_base import BaseModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self.name = ' ------- DeeperLSTM ------- '

        with self.g.as_default():
            self.sess = tf.Session()
            self.model = BaseModel(config, phase='test')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

        self._init_image_cache()

    def _init_image_cache(self):
        from util import load_hdf5
        d = load_hdf5('data/res152_std_mscoco_kptest.data')
        # d = load_hdf5('data/res152_std_mscoco_kpval.data')
        image_ids = d['image_ids']
        self.im_feats = d['features']
        image_id2index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.image_id2index = image_id2index

    def _load_image(self, image_id):
        idx = self.image_id2index[image_id]
        return self.im_feats[idx][np.newaxis, :]


class VQALoop(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.image_id = None
        # self.model = AttentionModel()
        self.models = [AttentionModel(), VanillaModel()]

    def do_set(self, image_id):
        self.image_id = int(image_id)

    def do_predict(self, question):
        print('Question:\n%s' % question)
        for model in self.models:
            model.inference(self.image_id, question)

    def do_p(self, question):
        self.do_predict(question)

    def do_r(self, question):
        self.do_predict(question)

    def do_EOF(self, line):
        return True

    def do_exit(self, line):
        exit(1)


if __name__ == '__main__':
    VQALoop().cmdloop()
