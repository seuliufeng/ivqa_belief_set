import numpy as np
from util import load_json, load_hdf5, save_json, save_hdf5, find_image_id_from_fname
from build_vqa_standard_data import AnswerEncoder
from w2v_answer_encoder import MultiChoiceQuestionManger


def get_image_id(split='train'):
    assert (split in ['train', 'val', 'test'])
    meta_path = '/homes/fl302/Downloads/caption_datasets/dataset_coco.json'
    d = load_json(meta_path)
    annotations = d['images']
    image_ids = [info['cocoid'] for info in annotations
                 if info['split'] == split]
    return image_ids


def concat_var_length_sequences(arr1, arr2):
    n_arr1, max_d_arr1 = arr1.shape
    n_arr2, max_d_arr2 = arr2.shape
    if max_d_arr1 != max_d_arr2:
        max_d = max(max_d_arr1, max_d_arr2)
        pad_d1 = max_d - max_d_arr1
        pad_d2 = max_d - max_d_arr2
        # padding
        pad_1 = np.zeros([n_arr1, pad_d1], dtype=arr1.dtype)
        arr1 = np.concatenate([arr1, pad_1], 1)
        pad_2 = np.zeros([n_arr2, pad_d2], dtype=arr2.dtype)
        arr2 = np.concatenate([arr2, pad_2], 1)
    # concatenate
    return np.concatenate([arr1, arr2], 0)


# def _load_dataset():
#     # load trainval
#     subset = 'trainval'
#     vqa_meta_file = 'data/vqa_std_mscoco_%s.meta' % subset
#     vqa_meta_trainval = load_json(vqa_meta_file)
#     vqa_data_file = 'data/vqa_std_mscoco_%s.data' % subset
#     vqa_data_trainval = load_hdf5(vqa_data_file)
#
#     attr_data_file = 'data/attribute_std_mscoco_%s.data' % subset
#     attr_data_trainval = load_hdf5(attr_data_file)
#
#     # load dev
#     subset = 'dev'
#     vqa_meta_file = 'data/vqa_std_mscoco_%s.meta' % subset
#     vqa_meta_dev = load_json(vqa_meta_file)
#     vqa_data_file = 'data/vqa_std_mscoco_%s.data' % subset
#     vqa_data_dev = load_hdf5(vqa_data_file)
#
#     attr_data_file = 'data/attribute_std_mscoco_%s.data' % subset
#     attr_data_dev = load_hdf5(attr_data_file)
#
#     # merge vqa data together
#     images = vqa_meta_trainval['images'] + vqa_meta_dev['images']
#     quest_id = vqa_meta_trainval['quest_id'] + vqa_meta_dev['quest_id']
#     quest_arr = concat_var_length_sequences(vqa_data_trainval['quest_arr'],
#                                             vqa_data_dev['quest_arr'])
#     quest_len = np.concatenate([vqa_data_trainval['quest_len'],
#                                 vqa_data_dev['quest_len']], 0)
#     answer = np.concatenate([vqa_data_trainval['answer'],
#                              vqa_data_dev['answer']], 0)
#
#     # merge attribute data
#     attr_image_ids = np.concatenate([attr_data_trainval['image_ids'], attr_data_dev['image_ids']], 0)
#     attr_arr = np.concatenate([attr_data_trainval['att_arr'], attr_data_dev['att_arr']], 0)
#     return images, quest_id, quest_arr, quest_len, answer, attr_image_ids, attr_arr


def _load_dataset():
    # load trainval
    subset = 'trainval'
    vqa_meta_file = '../iccv_vaq/data/vqa_std_mscoco_%s.meta' % subset
    vqa_meta_trainval = load_json(vqa_meta_file)

    # load dev
    subset = 'dev'
    vqa_meta_file = '../iccv_vaq/data/vqa_std_mscoco_%s.meta' % subset
    vqa_meta_dev = load_json(vqa_meta_file)

    return vqa_meta_trainval['quest_id'], vqa_meta_dev['quest_id']


def split_subset(subset, inputs):
    print('Processing split %s' % subset)
    images, quest_id, quest_arr, quest_len, answer, attr_image_ids, attr_arr = inputs
    vqa_image_ids = [find_image_id_from_fname(fname) for fname in images]
    # get coco ids
    coco_ids = get_image_id(subset)
    # build coco id hashing table
    coco_ids = {image_id: i for i, image_id in enumerate(coco_ids)}

    # split vqa data
    keep_tab = np.array([im_id in coco_ids for im_id in vqa_image_ids])
    images = [im for im, keep in zip(images, keep_tab) if keep]
    quest_id = [q_id for q_id, keep in zip(quest_id, keep_tab) if keep]
    quest_arr = quest_arr[keep_tab]
    quest_len = quest_len[keep_tab]
    answer = answer[keep_tab]

    # split attribute data
    keep_tab = np.array([im_id in coco_ids for im_id in attr_image_ids])
    attr_image_ids = attr_image_ids[keep_tab]
    attr_arr = attr_arr[keep_tab]

    # process answers
    encode_answers(quest_id, subset)

    # save to files
    # vqa_meta_file = 'data/vqa_std_mscoco_kp%s.meta' % subset
    # save_json(vqa_meta_file, {'images': images, 'quest_id': quest_id})
    #
    # vqa_data_file = 'data/vqa_std_mscoco_kp%s.data' % subset
    # save_hdf5(vqa_data_file, {'quest_arr': quest_arr, 'quest_len': quest_len,
    #                           'answer': answer})
    # attr_data_file = 'data/attribute_std_mscoco_kp%s.data' % subset
    # save_hdf5(attr_data_file, {'image_ids': attr_image_ids,
    #                            'att_arr': attr_arr})


def encode_answers(quest_ids, subset):
    ctx = MultiChoiceQuestionManger(subset='trainval', load_ans=True,
                                    answer_coding='sequence')
    answers = []
    for q_id in quest_ids:
        ans, seq = ctx.get_gt_answer_and_sequence_coding(q_id)
        answers += seq
    # merge questions to a matrix
    seq_len = [len(q) for q in answers]
    max_len = max(seq_len)
    num_capts = len(answers)
    ans_arr = np.zeros([num_capts, max_len], dtype=np.int32)
    for i, x in enumerate(ans_arr):
        x[:seq_len[i]] = answers[i]
    seq_len = np.array(seq_len, dtype=np.int32)
    vqa_data_file = 'data/answer_std_mscoco_%s.data' % subset
    save_hdf5(vqa_data_file, {'ans_arr': ans_arr, 'ans_len': seq_len})


def main():
    trainval_quest_ids, dev_quest_ids = _load_dataset()
    encode_answers(trainval_quest_ids, 'trainval')
    encode_answers(dev_quest_ids, 'dev')
    # split_subset('train', outputs)
    # split_subset('val', outputs)
    # split_subset('test', outputs)


if __name__ == '__main__':
    main()
