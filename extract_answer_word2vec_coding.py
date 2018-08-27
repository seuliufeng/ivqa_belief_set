import numpy as np
from w2v_answer_encoder import MultiChoiceQuestionManger
from util import load_hdf5, save_hdf5, load_json


class FeatureEncoder(object):
    def __init__(self, subset):
        data_file = 'data/res152_std_mscoco_%s.data' % subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        self.data = d['features']
        self.image_id2index = {image_id: i for i, image_id in enumerate(image_ids)}

    def get_feature(self, image_id):
        idx = self.image_id2index[image_id]
        return self.data[idx, :]


def load_dataset(subset):
    fpath = 'data/vqa_std_mscoco_%s.meta' % subset
    return load_json(fpath)['quest_id']


def process_dataset(mc, subset):
    print('Processing %s' % subset)
    quest_ids = load_dataset(subset)
    im_encoder = FeatureEncoder(subset)
    answer_enc = []
    image_enc = []
    for quest_id in quest_ids:
        _, w2v = mc.get_gt_answer_and_word2vec(quest_id)
        answer_enc.append(w2v)
        im_feat = im_encoder.get_feature(mc.get_image_id(quest_id))
        image_enc.append(im_feat[np.newaxis, :])
    quest_ids = np.array(quest_ids, dtype=np.int32)
    answer_enc = np.concatenate(answer_enc, axis=0).astype(np.float32)
    image_enc = np.concatenate(image_enc, axis=0).astype(np.float32)
    save_hdf5('data/image_answer_coding_%s.h5' % subset, {'quest_ids': quest_ids,
                                                          'answer_enc': answer_enc,
                                                          'image_enc': image_enc})


def main(_):
    mc = MultiChoiceQuestionManger(subset='trainval', answer_coding='word2vec')
    # process_dataset(mc, 'kptrain')
    # process_dataset(mc, 'kpval')
    process_dataset(mc, 'kptest')


if __name__ == '__main__':
    main(None)