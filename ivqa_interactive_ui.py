import tensorflow as tf
import numpy as np
import os
import cmd
from inference_utils.question_generator_util import SentenceGenerator
from util import load_json, load_hdf5
from nltk.tokenize import word_tokenize
from inference_utils import vocabulary
from post_process_variation_questions import post_process_variation_questions_with_count_v2


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


class SentenceEncoder(object):
    def __init__(self, type='answer'):
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


class VarIVQAModelWrapper(object):
    def __init__(self):
        self.to_sentence = SentenceGenerator(trainset='trainval')
        self.sent_encoder = SentenceEncoder()
        self.g = tf.Graph()
        self.ckpt_file = 'model/v1_var_kptrain_VAQ-VarDS/model.ckpt-3300000'
        from models.variational_ds_ivqa_model import VariationIVQAModel
        from config import ModelConfig
        config = ModelConfig()
        self._top_k = 10
        self.name = ' ------- VarIVQA ------- '

        with self.g.as_default():
            self.sess = tf.Session()
            self.model = VariationIVQAModel(config, phase='sampling_beam')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, self.ckpt_file)

        self._init_image_cache()

    def _init_image_cache(self):
        from util import load_hdf5
        d = load_hdf5('data/attribute_std_mscoco_kpval.data')
        # d = load_hdf5('data/res152_std_mscoco_kpval.data')
        image_ids = d['image_ids']
        self.im_feats = d['att_arr']
        image_id2index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.image_id2index = image_id2index

    def _load_image(self, image_id):
        idx = self.image_id2index[image_id]
        return self.im_feats[idx][np.newaxis, :]

    def _process_answer(self, answers):
        arr, arr_len = self.sent_encoder.encode_sentence(answers)
        return arr, arr_len

    def inference(self, image_id, answer):
        image = self._load_image(image_id)
        arr, arr_len = self._process_answer(answer)
        scores, pathes = self.model.greedy_inference([image, arr, arr_len], self.sess)
        self.show_prediction(scores, pathes)
        return scores

    def show_prediction(self, scores, pathes):
        # wrap inputs
        _this_batch_size = 1
        seq_len = pathes.shape[1]
        dummy_scores = np.tile(scores[:, np.newaxis], [1, seq_len])
        # dummy_scores = np.zeros_like(pathes, dtype=np.float32)
        ivqa_scores, ivqa_pathes, ivqa_counts = post_process_variation_questions_with_count_v2(dummy_scores, pathes,
                                                                                               _this_batch_size)
        for _q_idx, (ps, scs, cs) in enumerate(zip(ivqa_pathes, ivqa_scores, ivqa_counts)):
            inds = np.argsort(-np.array(scs))[:self._top_k]
            for _p_idx, _pick_id in enumerate(inds):
                _p = ps[_pick_id]
                _s = scs[_pick_id]  # / (len(_p) - 2)
                sentence = self.to_sentence.index_to_question(_p)
                print('%s (%0.2f)' % (sentence, _s))
                # import pdb
                # pdb.set_trace()


class IVQALoop(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.image_id = None
        self.model = VarIVQAModelWrapper()

    def do_set(self, image_id):
        self.image_id = int(image_id)

    def do_predict(self, ans):
        print('Answer: %s' % ans)
        self.model.inference(self.image_id, ans)
        print('\n')

    def do_p(self, ans):
        self.do_predict(ans)

    def do_g(self, ans):
        self.do_predict(ans)

    def do_r(self, question):
        self.do_predict(question)

    def do_EOF(self, line):
        return True

    def do_exit(self, line):
        exit(1)


if __name__ == '__main__':
    IVQALoop().cmdloop()
