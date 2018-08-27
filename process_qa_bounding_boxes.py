from util import load_json, save_hdf5
import os
import numpy as np
from PIL import Image
from build_vg_standard_data import _get_vg_image_root
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
IMFORMAT = '%s/%s'
IM_ROOT = '/usr/data/fl302/data/visual_genome'


def _get_image_size(im_file):
    img = Image.open(im_file)
    width, height = img.size
    return float(width), float(height)


def parse_regions():
    dataset = load_json('/usr/data/fl302/data/visual_genome/region_descriptions.json')

    regions = {}
    num = len(dataset)
    for i, info in enumerate(dataset):
        if i % 1000 == 0:
            tf.logging.info('Parse boxes: %d/%d' % (i, num))
        image_id = info['id']
        filename = IMFORMAT % _get_vg_image_root(image_id)
        im_w, im_h = _get_image_size(os.path.join(IM_ROOT, filename))

        for r in info['regions']:
            _region_id = r['region_id']
            _x = r['x'] / im_w
            _y = r['y'] / im_h
            _h = r['height'] / im_h
            _w = r['width'] / im_w
            box = [_y, _x, _y + _h, _x + _w]
            regions[_region_id] = box
    return regions


def _load_visual_7w_qa2image_ids():
    anno_path = '/usr/data/fl302/data/visual_genome/dataset_v7w_telling.json'
    dataset = load_json(anno_path)['images']

    qa_id2image_id = {}
    for info in dataset:
        for qa in info['qa_pairs']:
            quest_id = qa['qa_id']
            image_id = qa['image_id']
            qa_id2image_id[quest_id] = image_id
    return qa_id2image_id


def load_visual_7w_region_annotations():
    qa_id2image_id = _load_visual_7w_qa2image_ids()  # QA id to Image ID
    dataset = load_json('/usr/data/fl302/data/visual_genome/'
                        'v7w_telling_answers.json')['boxes']

    qid2box = {}
    num = len(dataset)
    for i, qa in enumerate(dataset):
        if i % 1000 == 0:
            tf.logging.info('Parse V7W boxes: %d/%d' % (i, num))

        qa_id = qa['qa_id']
        image_id = qa_id2image_id[qa_id]
        filename = IMFORMAT % _get_vg_image_root(image_id)
        im_w, im_h = _get_image_size(os.path.join(IM_ROOT, filename))

        _x = qa['x'] / im_w
        _y = qa['y'] / im_h
        _h = qa['height'] / im_h
        _w = qa['width'] / im_w
        box = [_y, _x, _y + _h, _x + _w]
        qid2box[qa_id] = box
    return qid2box


def find_boxes_of_questions(regions, v7w_qa2box, subset):
    seed_file = os.path.join('/usr/data/fl302/code/inverse_vqa/data2',
                             'v7w_std_mscoco_%s.meta' % subset)
    d = load_json(seed_file)
    quest_ids = d['quest_id']

    sv_file = '/usr/data/fl302/code/inverse_vqa/data2/v7w_qa_boxes_%s.data' % subset
    qa2reg = load_json('/usr/data/fl302/data/visual_genome/qa_to_region_mapping.json')
    has_boxes = []
    quest_boxes = []
    for quest_id in quest_ids:
        # see whether it is in v7w annotation
        if quest_id in v7w_qa2box:
            box = v7w_qa2box[quest_id]
            quest_boxes.append(box)
            has_boxes.append(True)
            continue
        # check visual genome annotation
        q_key = str(quest_id)
        if q_key in qa2reg:
            region_id = qa2reg[q_key]
            if region_id in regions:
                # print('In')
                box = regions[region_id]
                quest_boxes.append(box)
                has_boxes.append(True)
            else:
                box = [0., 0., 1., 1.]
                has_boxes.append(False)
                quest_boxes.append(box)
        else:
            box = [0., 0., 1., 1.]
            has_boxes.append(False)
            quest_boxes.append(box)
    quest_boxes = np.array(quest_boxes)
    has_boxes = np.array(has_boxes)
    quest_ids = np.array(quest_ids)
    tf.logging.info('Subset %s, %d/%d QAs have region annotation' % (subset,
                                                                     has_boxes.sum(),
                                                                     has_boxes.size))
    save_hdf5(sv_file, {'quest_ids': quest_ids, 'has_boxes': has_boxes,
                        'quest_boxes': quest_boxes})


if __name__ == '__main__':
    regions = parse_regions()
    v7w_qa2box = load_visual_7w_region_annotations()
    find_boxes_of_questions(regions, v7w_qa2box, 'train')
    find_boxes_of_questions(regions, v7w_qa2box, 'val')
    find_boxes_of_questions(regions, v7w_qa2box, 'test')
    find_boxes_of_questions(regions, v7w_qa2box, 'vg_aug_train')
