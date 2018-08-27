from vqa_interactive_ui import BaseVQAModel
import tensorflow as tf
import numpy as np
import os


class VanillaModel(BaseVQAModel):
    def __init__(self, ckpt_file='model/kprestval_VQA-BaseNorm/model.ckpt-26000'):
        BaseVQAModel.__init__(self)
        self.top_k = 2
        self.g = tf.Graph()
        self.ckpt_file = ckpt_file
        from models.vqa_base import BaseModel
        from vqa_config import ModelConfig
        config = ModelConfig()
        self._subset = 'test'
        self._year = 2015
        self.name = ' ------- DeeperLSTM ------- '

        with self.g.as_default():
            self.sess = tf.Session()
            self.model = BaseModel(config, phase='test')
            self.model.build()
            vars = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=vars)
            self.saver.restore(self.sess, ckpt_file)

    def _load_image(self, image_id):
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        filename = '%s%d/COCO_%s%d_%012d.jpg' % (self._subset, self._year,
                                                 self._subset, self._year, image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']

        f1 = np.mean(np.mean(f.transpose((1, 2, 0)), axis=0), axis=0)
        # f2 = f.reshape([2048, -1]).mean(axis=1)
        # import pdb
        # pdb.set_trace()
        return f1[np.newaxis, ::]
