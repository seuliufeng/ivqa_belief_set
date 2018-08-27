import os
import numpy as np
import pdb
from util import load_hdf5, load_json, find_image_id_from_fname, save_hdf5

_DATA_ROOT = './'


def load_qa_data(subset):
    # load meta file
    meta_file = os.path.join(_DATA_ROOT,
                             'data/vqa_std_mscoco_%s.meta' % subset)
    d = load_json(meta_file)
    image_ids = [find_image_id_from_fname(im_name) for im_name in d['images']]
    image_id2qa_index = {}
    for i, image_id in enumerate(image_ids):
        if image_id in image_id2qa_index:
            image_id2qa_index[image_id].append(i)
        else:
            image_id2qa_index[image_id] = [i]

    data_file = os.path.join(_DATA_ROOT,
                             'data/vqa_std_mscoco_%s.data' % subset)
    d = load_hdf5(data_file)
    labels = d['answer']
    return image_id2qa_index, labels


def load_image_data(subset):
    # load images
    data_file = 'data/res152_std_mscoco_%s.data' % subset
    d = load_hdf5(data_file)
    image_ids = d['image_ids']
    feats = d['features']
    return feats, image_ids


def process(subset):
    print('Processing subset %s' % subset)
    disable_entries = [0, 1, 2000]
    feats, image_ids = load_image_data(subset)
    image_id2qa_index, labels = load_qa_data(subset)
    num = feats.shape[0]
    bin_labels = np.zeros((num, 2001), dtype=np.float32)
    for image_id, bow in zip(image_ids, bin_labels):
        ind = image_id2qa_index[image_id]
        _label = np.unique(labels[ind])
        bow[_label] = 1.0
    # disable meaningless entries
    bin_labels[:, disable_entries] = 0.
    sv_file = 'data/vqa_std_mscoco_multilabel_%s.data' % subset
    save_hdf5(sv_file, {'features': feats,
                        'labels': bin_labels,
                        'image_ids': image_ids})


if __name__ == '__main__':
    process('kpval')
    process('kptest')
    process('kprestval')
    process('kptrain')
