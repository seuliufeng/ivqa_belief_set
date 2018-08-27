import json
import os
from shutil import copyfile

IM_ROOT = '/usr/data/fl302/data/VQA/Images/val2014/'
DST_DIR = 'bs_images'


def load_json(fpath):
    return json.load(open(fpath, 'r'))


def save_json(fpath, d):
    json.dump(d, open(fpath, 'w'))


def load_belief_sets(method, sample_method, train_method):
    if sample_method == 'rand':
        res_file = 'result/bs_%s_final_%s.json' % (train_method, method)
    else:
        res_file = 'result/bs_%s_final_%s_BEAM.json' % (train_method, method)
    # if os.path.exists(res_file):
    # print('File %s:' % res_file)
    # return
    return load_json(res_file)


def copy_images(im_file, do_copy=True):
    im_path = os.path.join(IM_ROOT, im_file)
    im_name = os.path.basename(im_file)
    dst_path = os.path.join(DST_DIR, im_name)
    if (not os.path.exists(dst_path)) and do_copy:
        copyfile(dst=dst_path, src=im_path)
    return im_name


def process():
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
    bs = load_belief_sets('Vanilla', 'beam', 'SL')
    for item in bs:
        image = os.path.basename(item['image'])
        copy_images(image)


if __name__ == '__main__':
    process()
