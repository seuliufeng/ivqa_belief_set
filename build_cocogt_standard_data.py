import os
import numpy as np
from scipy.io import loadmat
from util import save_hdf5
# from build_caption_standard_data import get_subset_image_id
from data_packer.build_karpathy_split_data import get_image_id


def _split_attributes(subset, atts, image_ids):
    image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
    subset_image_ids = get_image_id(subset)
    index = -np.ones(len(subset_image_ids), dtype=np.int32)
    for i, image_id in enumerate(subset_image_ids):
        if image_id in image_id2att_index:
            index[i] = image_id2att_index[image_id]
    num_match = (index >= 0).sum()
    print('Find %d matched attributes for subset %s, missed %d\n' % (num_match,
                                                                     subset, index.size-num_match))
    # slice
    index[index < 0] = 0
    scores = atts[index, :]
    data_file = 'data/capt1k_std_mscoco_kp%s.data' % subset
    save_hdf5(data_file, {'att_arr': scores.astype(np.float32),
                          'image_ids': np.array(subset_image_ids, dtype=np.int32)})


def load_gt_attributes():
    data_root = '/import/vision-ephemeral/fl302/code/slim'
    subsets = ['Train', 'Val']
    image_ids, labels = [], []
    for subset in subsets:
        fname = 'Inception_v3_1000_%s_Dets.mat' % subset
        print('Loading %s...' % fname)
        fpath = os.path.join(data_root, fname)
        d = loadmat(fpath)
        t_labels = d['labels']
        t_image_ids = d['image_id'].flatten()
        image_ids.append(t_image_ids)
        labels.append(t_labels)
    image_ids = np.concatenate(image_ids).astype(np.int32)
    labels = np.concatenate(labels).astype(np.float32)
    return image_ids, labels


def convert_val_attributes():
    data_root = '/import/vision-ephemeral/fl302/code/slim'
    subsets = ['Val']
    image_ids, labels = [], []
    for subset in subsets:
        fname = 'Inception_v3_1000_%s_Dets.mat' % subset
        print('Loading %s...' % fname)
        fpath = os.path.join(data_root, fname)
        d = loadmat(fpath)
        t_labels = d['labels'].astype(np.float32)
        t_image_ids = d['image_id'].flatten()
        data_file = 'data/capt1k_std_mscoco_%s.data' % subset.lower()
        save_hdf5(data_file, {'att_arr': t_labels.astype(np.float32),
                              'image_ids': np.array(t_image_ids, dtype=np.int32)})




if __name__ == '__main__':
    convert_val_attributes()
    # image_ids, attributes = load_gt_attributes()
    # _split_attributes('train', attributes, image_ids)
    # _split_attributes('val', attributes, image_ids)
    # _split_attributes('test', attributes, image_ids)

