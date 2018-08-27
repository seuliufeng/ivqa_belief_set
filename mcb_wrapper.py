import os
from vqa_data_provider_layer import LoadVQADataProvider
import caffe
import numpy as np


import pdb

def _softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist


class MCBModel(object):
    def __init__(self):
        self.gpu_id = 0
        self.name = ' ------- MCB-attention ------- '
        model_root = '/usr/data/fl302/code/vqa-mcb/multi_att_2_glove_pretrained'
        vdict_path = os.path.join(model_root, 'vdict.json')
        adict_path = os.path.join(model_root, 'adict.json')
        EXTRACT_LAYER_SIZE = (2048, 14, 14)
        self.reader = LoadVQADataProvider(vdict_path,
                                          adict_path,
                                          batchsize=1,
                                          mode='test',
                                          data_shape=EXTRACT_LAYER_SIZE)
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        VQA_PROTOTXT_PATH = os.path.join(model_root, 'proto_test_batchsize1.prototxt')
        VQA_CAFFEMODEL_PATH = os.path.join(model_root, "_iter_190000.caffemodel")
        self.vqa_net = caffe.Net(VQA_PROTOTXT_PATH, VQA_CAFFEMODEL_PATH, caffe.TEST)

    def inference(self, image_id, question):
        self._load_image(image_id)
        self._process_question(question)
        scores = self.compute_scores()
        self.show_prediction(scores)

    def get_score(self, image_id, question):
        self._load_image(image_id)
        self._process_question(question)
        scores = self.compute_scores()
        # pdb.set_trace()
        scores = scores.flatten()
        id = scores.argmax()
        sc = scores[id]
        answer = self.reader.vec_to_answer(id)
        return answer, sc

    def _load_image(self, image_id):
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        filename = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
        f = (f / np.sqrt((f ** 2).sum()))
        # pdb.set_trace()
        self.vqa_net.blobs['img_feature'].data[...] = f[np.newaxis, ::]

    def _process_question(self, question):
        qvec, cvec, avec, glove_matrix = self.reader.create_batch(question)
        # pdb.set_trace()
        self.vqa_net.blobs['data'].data[...] = np.transpose(qvec, (1, 0))
        self.vqa_net.blobs['cont'].data[...] = np.transpose(cvec, (1, 0))

        self.vqa_net.blobs['label'].data[...] = avec  # dummy
        self.vqa_net.blobs['glove'].data[...] = np.transpose(glove_matrix, (1, 0, 2))

    def compute_scores(self):
        self.vqa_net.forward()
        scores = self.vqa_net.blobs['prediction'].data.flatten()
        # scores = self.vqa_net.blobs['prediction'].data
        # scores = scores.copy()[-1, :]
        # attention visualization
        # att_map = self.vqa_net.blobs['att_map0'].data.copy()[0]
        # source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')
        # path0 = save_attention_visualization(source_img_path, att_map, img_ques_hash)
        scores = _softmax(scores)
        return scores

    def show_prediction(self, scores):
        scores = scores.flatten()
        inds = (-scores).argsort()[:2]
        print('%s' % self.name)
        for id in inds:
            sc = scores[id]
            answer = self.reader.vec_to_answer(id)
            print('%s: %0.2f' % (answer, sc))


if __name__ == '__main__':
    model = MCBModel()
    model.inference(96549, 'Is this a fighter plane ?')
