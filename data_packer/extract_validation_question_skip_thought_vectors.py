import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

from util import *


def _read_json(data_root, filename, key=None):
    d = json.load(open(os.path.join(data_root,
                                    filename), 'r'))
    return d if key is None else d[key]


def main():
    ann_root = '/home/fl302/Projects/VQA-tensorflow/data/annotations'
    questions = _read_json(ann_root, 'MultipleChoice_mscoco_val2014_questions.json', 'questions')

    # create model
    model = skipthoughts.load_model()

    # create buffers
    quest_ids, quest_coding = [], []
    quest_buffer = []
    # now, do the job
    for i, info in enumerate(questions):
        print('Skip thought: extracted %d/%d' % (i, len(questions)))
        if i > 100:
            break
        quest_id = info['question_id']
        quest = info['question'].lower()
        quest_buffer.append(quest)
        quest_ids.append(quest_id)
        if i % 100 == 0 and i > 0:
            quest_vectors = skipthoughts.encode(model, quest_buffer)
            # append to the main buffer
            quest_coding.append(quest_vectors.copy())
            # clear question buffer
            quest_buffer = []
    # process last batch
    if quest_buffer:
        quest_vectors = skipthoughts.encode(model, quest_buffer)
        quest_coding.append(quest_vectors.copy())

    # concatenate
    quest_coding = np.concatenate(quest_coding, axis=0).astype(np.float32)
    quest_ids = np.array(quest_ids, dtype=np.int32)

    # save to file
    save_hdf5('vqa_val_skipthought.h5', {'quest_id': quest_ids,
                                         'quest_coding': quest_coding})


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--caption_file', type=str, default='Data/sample_captions.txt',
#                         help='caption file')
#     parser.add_argument('--data_dir', type=str, default='Data',
#                         help='Data Directory')
#
#     args = parser.parse_args()
#     with open(args.caption_file) as f:
#         captions = f.read().split('\n')
#
#     captions = [cap for cap in captions if len(cap) > 0]
#     print captions
#     model = skipthoughts.load_model()
#     caption_vectors = skipthoughts.encode(model, captions)
#
#     if os.path.isfile(join(args.data_dir, 'sample_caption_vectors.hdf5')):
#         os.remove(join(args.data_dir, 'sample_caption_vectors.hdf5'))
#     h = h5py.File(join(args.data_dir, 'sample_caption_vectors.hdf5'))
#     h.create_dataset('vectors', data=caption_vectors)
#     h.close()


if __name__ == '__main__':
    # load_VQA_validation_questions()
    main()
