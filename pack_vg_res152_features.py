import numpy as np
from time import time
import os
from util import save_hdf5, load_json


def _load_and_pool_feature(fname):
    feats = np.load(fname)['x']
    return feats.mean(axis=-1).mean(axis=-1)[np.newaxis, :]


def pool_resnet_features(split='train'):
    SPLITS = ['test', 'train', 'val', 'vg_aug_train']
    assert (split in SPLITS)
    seed_file = os.path.join('/usr/data/fl302/code/inverse_vqa/data2',
                             'v7w_std_mscoco_%s.meta' % split)
    d = load_json(seed_file)
    image_ids = d['image_ids']
    image_names = d['images']
    image_id2fpath = {}

    for image_id, name in zip(image_ids, image_names):
        image_id2fpath.update({image_id: name})

    FEAT_DIR = '/usr/data/fl302/data/visual_genome/ResNet152/resnet_res5c'

    image_ids = np.unique(image_ids)

    idx = 0
    t = time()
    feats = []
    for image_id in image_ids:
        if idx % 100 == 0:
            print('processed %d images (%0.2f sec/batch)' % (idx, time() - t))
            t = time()

        file_name = image_id2fpath[image_id]
        feat_file = os.path.join(FEAT_DIR, file_name + '.npz')
        f = _load_and_pool_feature(feat_file)
        feats.append(f)
        idx += 1
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    save_hdf5('data2/v7w_res152_%s.h5' % split, {'image_ids': image_ids,
                                                 'features': feats})


if __name__ == '__main__':
    pool_resnet_features('val')
    pool_resnet_features('test')
    pool_resnet_features('train')
    pool_resnet_features('vg_aug_train')
