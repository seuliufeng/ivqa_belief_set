import tensorflow as tf
import numpy as np
import os
from util import load_json, load_hdf5, find_image_id_from_fname, get_feature_root, save_json
from post_process_variation_questions import put_to_array
import pdb


class VQAData(object):
    def __init__(self, subset='kprestval'):
        self._subset = subset
        self._FEAT_ROOT = get_feature_root(self._subset, 'res5c')
        self._load_data()

    def _load_data(self):
        meta_file = 'data/vqa_std_mscoco_%s.meta' % self._subset
        data_file = 'data/vqa_std_mscoco_%s.data' % self._subset
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        self._quest_ids = np.array(d['quest_id'])
        self.quest_id2index = {qid: i for i, qid in enumerate(self._quest_ids)}
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)

        # load QA data
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['answer'].astype(np.int32)
        self._load_global_image_feature()

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2feat_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._feat = d['features']

    def _load_image_features(self, idx):
        filename = self._images[idx]
        f = np.load(os.path.join(self._FEAT_ROOT, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]

    def _load_attribute_feature(self, idx):
        _slice_row = self._vqa_index2feat_index[idx]
        return self._feat[_slice_row][np.newaxis, :]

    def get_data(self, quest_id):
        index = self.quest_id2index[quest_id]
        # images = self._load_image_features(index)
        images = self._load_attribute_feature(index)
        answer = self._answer[index]
        return images, answer


def process_questions(pathes):
    token_pathes = []
    for _p in pathes:
        tokens = [int(t) for t in _p.split(' ')]
        token_pathes.append(tokens[1:-1])
    return put_to_array(token_pathes)


def create_model():
    from models.vq_matching import BaseModel
    from vqa_config import ModelConfig
    model = BaseModel(ModelConfig(), phase='test_broadcast')
    model.build()

    checkpoint_path = 'model/kprestval_VQ-Match/model.ckpt-22000'
    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    return sess, model


def score_replay_buffer():
    d = load_json('vqa_replay_buffer/low/vqa_replay.json')
    vqa_data = VQAData()
    # vqa_thresh = 0.3
    vqa_thresh = 0.5
    # match_thresh = 0.2
    match_thresh = 0.4

    # create model
    sess, model = create_model()

    memory = d['memory']
    new_memory = {}
    for i, quest_key in enumerate(memory.keys()):
        pathes = memory[quest_key]
        if len(pathes) == 0:
            continue
        if i % 100 == 0:
            print('Processed %d/%d items' % (i, len(memory)))
        # if i > 1000:
        #     break
        new_memory[quest_key] = {}
        # if it has valid questions
        quest_id = int(quest_key)
        pathes = memory[quest_key].keys()
        quest, quest_len = process_questions(pathes)
        image, top_ans = vqa_data.get_data(quest_id)
        vqa_inputs = [image, quest, quest_len, np.zeros_like(quest_len)]
        # import pdb
        # pdb.set_trace()
        scores = sess.run(model.prob,
                          feed_dict=model.fill_feed_dict(vqa_inputs))
        # confs = scores[:, top_ans]
        confs = scores
        ref = memory[quest_key]
        for path_key, new_conf in zip(pathes, confs):
            _vqa_score = ref[path_key]
            _match_score = new_conf
            if _vqa_score > vqa_thresh and _match_score > match_thresh:
                new_memory[quest_key][path_key] = (ref[path_key], float(new_conf))
    # save
    save_json('vqa_replay_buffer/vqa_replay_low_rescore_prior_05_04.json',
              {'memory': new_memory})


if __name__ == '__main__':
    score_replay_buffer()
