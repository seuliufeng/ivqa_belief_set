import os
from vqa_data_provider_layer import LoadVQADataProvider
import caffe
import numpy as np
from config import VOCAB_CONFIG
# import pdb

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id


class MCBModel(object):
    def __init__(self, batch_size):
        self.gpu_id = 0
        self.name = ' ------- MCB-attention ------- '
        model_root = '/usr/data/fl302/code/vqa-mcb/multi_att_2_glove_pretrained'
        vdict_path = os.path.join(model_root, 'vdict.json')
        adict_path = os.path.join(model_root, 'adict.json')
        EXTRACT_LAYER_SIZE = (2048, 14, 14)
        self.batch_size = batch_size
        # Note: to change batch size, should modify, data layer, here, and dummy data
        # layer in prototext of attention layer
        self.reader = LoadVQADataProvider(vdict_path,
                                          adict_path,
                                          batchsize=self.batch_size,
                                          mode='test',
                                          data_shape=EXTRACT_LAYER_SIZE)
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        VQA_PROTOTXT_PATH = os.path.join(model_root, 'proto_test_batchsize_n.prototxt')
        VQA_CAFFEMODEL_PATH = os.path.join(model_root, "_iter_190000.caffemodel")
        self.vqa_net = caffe.Net(VQA_PROTOTXT_PATH, VQA_CAFFEMODEL_PATH, caffe.TEST)

    def inference(self, image_id, question):
        self._load_images(image_id)
        self._process_questions(question)
        scores = self.compute_scores()
        return scores
        # self.show_prediction(scores)

    def inference_batch(self, images, questions):
        self._feed_images(images)
        self._process_questions(questions)
        return self.compute_scores()

    def _load_images(self, image_ids):
        images = []
        FEAT_ROOT = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
        for image_id in image_ids:
            filename = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
            f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
            f = (f / np.sqrt((f ** 2).sum()))
            images.append(f[np.newaxis, ::])
        images = np.concatenate(images, axis=0)
        self._feed_images(images)

    def _feed_images(self, images):
        self.vqa_net.blobs['img_feature'].data[...] = images[...]  # assign by value

    def _process_questions(self, questions):
        qvec, cvec, avec, glove_matrix = self.reader.create_batch_multiple(questions)
        self.vqa_net.blobs['data'].data[...] = np.transpose(qvec, (1, 0))
        self.vqa_net.blobs['cont'].data[...] = np.transpose(cvec, (1, 0))

        self.vqa_net.blobs['label'].data[...] = avec  # dummy
        self.vqa_net.blobs['glove'].data[...] = np.transpose(glove_matrix, (1, 0, 2))

    def compute_scores(self):
        self.vqa_net.forward()
        scores = self.vqa_net.blobs['prob'].data.copy()
        # attention visualization
        # att_map = self.vqa_net.blobs['att_map0'].data.copy()[0]
        # source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')
        # path0 = save_attention_visualization(source_img_path, att_map, img_ques_hash)
        # scores = _softmax(scores)
        return scores

    def get_top_answer_ids(self, answers):
        top_ans_ids = []
        for ans in answers:
            top_ans_ids.append(self.reader.answer_to_vec(ans))
        return np.array(top_ans_ids, dtype=np.int32)

    def show_prediction(self, scores):
        scores = scores.flatten()
        inds = (-scores).argsort()[:2]
        print('%s' % self.name)
        for id in inds:
            sc = scores[id]
            answer = self.reader.vec_to_answer(id)
            print('%s: %0.2f' % (answer, sc))


class MCBReward(object):
    def __init__(self, to_sentence):
        self.to_sentence = to_sentence
        self.model = MCBModel(batch_size=80)

    def get_top_answer_ids(self, ans, ans_len):
        def _array_to_path(capt, capt_len):
            seqs = []
            for c, clen in zip(capt, capt_len):
                seqs.append(c[:clen])
            return seqs

        pathes = _array_to_path(ans, ans_len)
        answers = [self.to_sentence.index_to_answer(p) for p in pathes]
        return self.model.get_top_answer_ids(answers)

    def get_reward(self, sampled, inputs):
        images, res5c, ans, ans_len, _ = inputs
        images_aug = []
        top_ans_ids_aug = []
        answer_aug = []
        answer_len_aug = []
        pathes = []
        top_ans_ids = self.get_top_answer_ids(ans, ans_len)
        res5c_aug = []
        for _idx, (ps, f) in enumerate(zip(sampled, res5c)):
            _n = len(ps)
            f = (f / np.sqrt((f ** 2).sum()))
            f = f[np.newaxis, ::].transpose((0, 3, 1, 2))  # convert NxHxWxC to NxCxHxW
            res5c_aug.append(np.tile(f, [_n, 1, 1, 1]))

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
        is_valid = top_ans_ids_aug != 2999
        return vqa_scores, [images_aug, answer_aug, answer_len_aug, is_valid]


if __name__ == '__main__':
    model = MCBModel(batch_size=2)
    scores = model.inference([96549, 96549], ['Is this a fighter plane ?',
                                              'Is this a commercial plane?'])
    import pdb

    pdb.set_trace()
