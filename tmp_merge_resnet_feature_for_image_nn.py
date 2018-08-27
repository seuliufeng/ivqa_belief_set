import numpy as np
from util import load_feature_file_vqabaseline, save_hdf5


def load_data():
    subsets = ['train', 'val']
    image_ids, features = [], []
    for subset in subsets:
        print('Loading subset: %s' % subset)
        fpath = 'data/imagenet_%s_features.h5' % subset
        d = load_feature_file_vqabaseline(fpath)
        image_ids.append(d['image_ids'])
        features.append(d['features'])
    image_ids = np.concatenate(image_ids).astype(np.int32)
    features = np.concatenate(features).astype(np.float32)
    print('Saving...')
    save_hdf5('/usr/data/fl302/code/compute_nn/res152_trainval.h5', {'image_ids': image_ids,
                                                                     'features': features})
    print('Done')


if __name__ == '__main__':
    load_data()

