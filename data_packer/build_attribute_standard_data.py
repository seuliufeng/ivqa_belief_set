import os
import numpy as np
from scipy.io import loadmat
from util import save_hdf5
from build_caption_standard_data import get_subset_image_id


def _split_attributes(subset, atts, image_ids):
    image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
    subset_image_ids = get_subset_image_id(subset)
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
    data_file = 'data/attribute_std_mscoco_%s.data' % subset
    save_hdf5(data_file, {'att_arr': scores.astype(np.float32),
                          'image_ids': np.array(subset_image_ids, dtype=np.int32)})


if __name__ == '__main__':
    data_dir = '/import/vision-ephemeral/fl302/data/share/sRNN/data'
    # att_file = 'Inception_v3_1000_Trainval_Dets.mat'
    att_file = 'Inception_v3_1000_Test_Dets.mat'
    d = loadmat(os.path.join(data_dir, att_file))
    if 'scores' in d:
        attributes = d['scores'].astype(np.float32)
        image_ids = d['image_id'].astype(np.int32).flatten()
    else:
        attributes = d['probs'].astype(np.float32)
        image_ids = d['index'].astype(np.int32).flatten()
    # _split_attributes('trainval', attributes, image_ids)
    # _split_attributes('dev', attributes, image_ids)
    _split_attributes('test-dev', attributes, image_ids)
    _split_attributes('test', attributes, image_ids)

