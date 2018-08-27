import os
import numpy as np
from util import load_hdf5, save_hdf5



def get_image_ids():
    seed_path = 'data/capt1k_std_mscoco_val.data'
    d = load_hdf5(seed_path)
    return d['image_ids']


def load_res152_feature():
    sets = ['val', 'test', 'restval']
    fdir = '/import/vision-ephemeral/fl302/code/text-to-image/'
    feats = []
    image_ids = []
    for subset in sets:
        d = load_hdf5(os.path.join(fdir, 'mscoco_res152_%s.h5' % subset))
        image_ids.append(d['image_ids'].flatten())
        feats.append(d['features'])
    feats = np.concatenate(feats)
    image_ids = np.concatenate(image_ids)

    # vertify
    vertify_image_ids(image_ids)

    # save
    data_file = 'data/res152_std_mscoco_%s.data' % 'val'
    save_hdf5(data_file, {'att_arr': feats.astype(np.float32),
                          'image_ids': np.array(image_ids, dtype=np.int32)})


def vertify_image_ids(image_ids):
    ref_image_ids = get_image_ids()
    ref_image_ids.sort()
    tar_image_ids = image_ids.copy()
    tar_image_ids.sort()
    is_valid = (tar_image_ids == ref_image_ids).sum() == tar_image_ids.size
    if is_valid:
        print ('Passed')
    else:
        raise Exception('verfication failed')


if __name__ == '__main__':
    load_res152_feature()

