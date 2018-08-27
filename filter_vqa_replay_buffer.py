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

    def _load_image_features(self, idx):
        filename = self._images[idx]
        f = np.load(os.path.join(self._FEAT_ROOT, filename + '.npz'))['x']
        return f.transpose((1, 2, 0))[np.newaxis, ::]

    def get_data(self, quest_id):
        index = self.quest_id2index[quest_id]
        images = self._load_image_features(index)
        answer = self._answer[index]
        return images, answer


def process_questions(pathes):
    token_pathes = []
    for _p in pathes:
        tokens = [int(t) for t in _p.split(' ')]
        token_pathes.append(tokens[1:-1])
    return put_to_array(token_pathes)


def create_model():
    from models.vqa_soft_attention import AttentionModel
    from vqa_config import ModelConfig
    model = AttentionModel(ModelConfig(), phase='test_broadcast')
    model.build()

    checkpoint_path = 'model/v1_vqa_VQA/v1_vqa_VQA_best2/model.ckpt-135000'
    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    return sess, model


def score_replay_buffer():
    d = load_json('vqa_replay_buffer/low/vqa_replay.json')
    vqa_data = VQAData()

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
        new_memory[quest_key] = {}
        # if it has valid questions
        quest_id = int(quest_key)
        pathes = memory[quest_key].keys()
        quest, quest_len = process_questions(pathes)
        image, top_ans = vqa_data.get_data(quest_id)
        vqa_inputs = [image, quest, quest_len, top_ans]
        scores = sess.run(model.prob,
                          feed_dict=model.fill_feed_dict(vqa_inputs))
        confs = scores[:, top_ans]
        ref = memory[quest_key]
        for path_key, new_conf in zip(pathes, confs):
            new_memory[quest_key][path_key] = (ref[path_key], float(new_conf))
    # save
    save_json('vqa_replay_buffer/vqa_replay_low_rescore.json',
              {'memory': new_memory})


if __name__ == '__main__':
    score_replay_buffer()
