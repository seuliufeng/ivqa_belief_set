import os
import numpy as np
from util import load_hdf5, save_hdf5


def load_data():
    feat_root = '/import/vision-datasets001/fl302/code/inverse_vqa/data'
    data_split = ['kptrain', 'val']
    features, image_ids = [], []
    for sp in data_split:
        fname = 'res152_std_mscoco_%s.data' % sp
        d = load_hdf5(os.path.join(feat_root, fname))
        image_ids.append(d['image_ids'])
        try:
            features.append(d['features'])
        except:
            features.append(d['att_arr'])
    image_ids = np.concatenate(image_ids)
    features = np.concatenate(features)
    num = image_ids.size
    print('Loaded %d images' % num)
    image_id2index = {im_id: i for i, im_id in enumerate(image_ids)}
    return image_ids, features, image_id2index


def split_data():
    seed_root = '../iccv_vaq/data'
    data_split = ['trainval', 'dev']
    seed_format = 'attribute_std_mscoco_%s.data'

    image_ids, features, image_id2index = load_data()
    for sp in data_split:
        print('Processing split: %s' % sp)
        seed_file = os.path.join(seed_root, seed_format % sp)
        d = load_hdf5(seed_file)
        sp_image_ids = d['image_ids']
        sp_order = []
        for image_id in sp_image_ids:
            sp_order.append(image_id2index[image_id])
        sp_features = features[sp_order]
        sv_file = os.path.join(seed_root, 'res152_std_mscoco_%s.data' % sp)
        save_hdf5(sv_file, {'image_ids': sp_image_ids, 'features': sp_features})


if __name__ == '__main__':
    split_data()
