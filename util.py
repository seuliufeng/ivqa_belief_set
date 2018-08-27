import tensorflow as tf
import cPickle
import os
from time import sleep
from datetime import datetime
import numpy as np
import h5py
import json
import sys
import re
from rnn_compact_ops import *


def update_progress(progress):
    barLength = 20
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rFinshed Percent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), int(progress * 100),
                                                      status)
    sys.stdout.write(text)
    sys.stdout.flush()


def save_hdf5(fname, d):
    hf = h5py.File(fname, 'w')
    for key in d.keys():
        value = d[key]
        hf.create_dataset(key,
                          dtype=value.dtype.name,
                          data=value)
    hf.close()
    return fname


def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


def load_json(fpath):
    return json.load(open(fpath, 'r'))


def save_json(fpath, d):
    json.dump(d, open(fpath, 'w'))


def create_init_variables_op(model_path, tvars):
    d = np.load(model_path).item()
    init_vars = []
    init_ops = []
    for var in tvars:
        for key in d:
            if key in var.name:
                tf.logging.info('initialize variable %s' % var.name)
                init_ops.append(var.assign(d[key]))
                init_vars.append(var.name)
    for vname in init_vars:
        tf.logging.info('Loaded variable %s' % vname)
    tf.logging.info('Done!')
    return init_ops


def get_default_initializer():
    return tf.random_uniform_initializer(-0.08, 0.08, dtype=tf.float32)


def create_dropout_lstm_cells(n_cells, input_keep_prob=1.0, output_keep_prob=1.0, state_is_tuple=True):
    lstm = LSTMCell(n_cells, initializer=get_default_initializer(), state_is_tuple=state_is_tuple)
    return DropoutWrapper(lstm, input_keep_prob, output_keep_prob)


def create_dropout_basic_lstm_cells(n_cells, input_keep_prob=1.0, output_keep_prob=1.0, state_is_tuple=True):
    lstm = BasicLSTMCell(num_units=n_cells, state_is_tuple=state_is_tuple)
    return DropoutWrapper(lstm, input_keep_prob, output_keep_prob)


def pickle(fname, data):
    with open(fname, 'wb') as pkl_file:
        cPickle.dump(data, pkl_file)


def unpickle(fname):
    with open(fname, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def decode_raw_format_sample(datum, feat_shape):
    datum = tf.decode_raw(datum, out_type=tf.float32)
    return tf.reshape(datum, feat_shape)


def decode_raw_format_sample_trans(datum, feat_shape):
    datum = tf.decode_raw(datum, out_type=tf.float32)
    datum = tf.reshape(datum, feat_shape)
    return tf.transpose(datum, perm=[1, 2, 0])


def decode_jpg_image(datum, image_mean):
    image = tf.image.decode_jpeg(datum, channels=3)
    return tf.cast(image, dtype=np.float32) - image_mean


def get_dataset_root():
    if os.path.exists('/usr/data/fl302/data/VQA'):
        data_root = '/usr/data/fl302/data/VQA'
        toolkit_root = '/usr/data/fl302/toolbox/coco-caption-master/'
    elif os.path.exists('/import/vision-datasets001/fl302/code/VQA'):
        data_root = '/import/vision-datasets001/fl302/code/VQA'
        toolkit_root = '/homes/fl302/projects/coco-caption'
    else:
        data_root = '/data/fl302/data/VQA'
        toolkit_root = '/data/fl302/data/im2txt/coco-caption'
    return data_root, toolkit_root


def get_model_iteration(model_path):
    import re
    model_name = os.path.basename(model_path)
    digits = ''.join(re.findall(r'\d+', model_name))
    return int(digits)


def wait_to_start(file):
    while True:
        if not os.path.exists(file):
            fname = os.path.basename(file)
            print('%s Waitting [%s] to start' % (datetime.now(), fname))
            sleep(60)
        else:
            print('Easy, file detected, will start in 500 sec...')
            sleep(500)
            break


def find_image_id_from_fname(filename):
    return int(re.findall('\d+', filename)[-1])


def get_res5c_feature_root(subset='dev'):
    # if 'test' in subset:
    #     return '/import/vision-ephemeral/fl302/data/VQA/ResNet152/resnet_res5c'
    if os.path.exists('/usr/data/fl302/data/VQA/ResNet152/'):
        feat_root = '/usr/data/fl302/data/VQA/ResNet152/resnet_res5c'
    elif os.path.exists('/home/fl302/data/ResNet152/resnet_res5c'):
        feat_root = '/home/fl302/data/ResNet152/resnet_res5c'
    elif os.path.exists('data/resnet_res5c'):
        feat_root = 'data/resnet_res5c'
    else:
        print('Can''t find res_5c features')
    return feat_root


def get_inception_feature_root(subset='dev'):
    if os.path.exists('/scratch/fl302/Inception/Logits'):
        return '/scratch/fl302/Inception/Logits'
    elif os.path.exists('/data/fl302/data/VQA/Inception/Logits/'):
        return '/data/fl302/data/VQA/Inception/Logits/'
    else:
        return '/import/vision-ephemeral/fl302/data/VQA/Inception/Logits'


def get_mix7c_feature_root(subset='dev'):
    if os.path.exists('/import/vision-datasets001/fl302/features/Inception/Mixed7c'):
        return '/import/vision-datasets001/fl302/features/Inception/Mixed7c'
    else:
        return '/data/fl302/data/VQA/Inception/Mixed7c'


def get_image_feature_root(subset='dev'):
    if os.path.exists('/scratch/fl302/VQA/Images'):
        return '/scratch/fl302/VQA/Images'
    elif os.path.exists('/import/vision-ephemeral/fl302/data/VQA/Images/mscoco'):
        return '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco'
    else:
        return '/data/fl302/data/mscoco/raw-data'


def get_feature_root(subset, feat_type='res5c'):
    feat_type = feat_type.lower()
    if feat_type == 'res5c':
        return get_res5c_feature_root(subset)
    elif feat_type == 'inception':
        return get_inception_feature_root(subset)
    elif feat_type == 'mix7c':
        return get_mix7c_feature_root()
    elif feat_type == 'image':
        return get_image_feature_root()
    return ''


def load_feature_file_vqabaseline(fpath):
    fs = h5py.File(fpath, 'r')
    tmp = fs['data']
    features = np.array(tmp['features'], dtype=np.float32)
    imageid = np.array(tmp['imageid'], dtype=np.int32).flatten()
    fs.close()
    return {'image_ids': imageid, 'features': features}
