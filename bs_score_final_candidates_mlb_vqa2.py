from time import time
from util import load_json, save_json
# from n2mn_wrapper import N2MNWrapper
import tensorflow as tf
import os
import numpy as np
from inference_utils.question_generator_util import SentenceGenerator
from vqa_interactive_ui import BaseVQAModel
from w2v_answer_encoder import MultiChoiceQuestionManger


class AttentionModel(BaseVQAModel):
    def __init__(self, subset='val'):
        BaseVQAModel.__init__(self)
        model_dir = '/usr/data/fl302/code/inverse_vqa/model/mlb_attention_v2/'
        top_ans_file = model_dir + 'vqa_trainval_top2000_answers.txt'
        ckpt_file = model_dir + 'model.ckpt-170000'
        self.to_sentence = SentenceGenerator(trainset='trainval',
                                             top_ans_file=top_ans_file)
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        from models.vqa_soft_attention_v2 import AttentionModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self._subset = subset
        self._year = 2015 if subset == 'test' else 2014
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
        filename = '%s%d/COCO_%s%d_%012d.jpg' % (self._subset, self._year,
                                                 self._subset, self._year, image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]


def load_results():
    return load_json('result/samples_to_score.json')


def compare_answer(a1, a2):
    return a1.lower().strip() == a2.lower().strip()


def process(model_type='mlb_v2'):
    cands = load_results()
    model = AttentionModel()
    mc_ctx = MultiChoiceQuestionManger(subset='val')

    results = {}
    t = time()
    print('Number items: %d' % len(cands))
    for i, res_key in enumerate(cands):
        if i % 100 == 0:
            avg_time = (time() - t) / 100.
            print('%d/%d (%0.2f sec/sample)' % (i, len(cands), avg_time))
            t = time()
        res_i = cands[res_key]
        image_id = res_i['image_id']
        question = res_i['target']
        question_id = res_i['question_id']
        gt_answer = mc_ctx.get_gt_answer(question_id)
        pred_ans, scores = model.get_score(image_id, question)
        sc = float(scores)
        is_valid = compare_answer(pred_ans, gt_answer)
        # if not is_valid:
        #     continue
        results[res_key] = {'pred_answer': pred_ans,
                            'pred_score': sc,
                            'gt_answer': gt_answer,
                            'is_valid': is_valid}
    save_json('result/%s_scores_final_v2.json' % model_type, results)


if __name__ == '__main__':
    process()
