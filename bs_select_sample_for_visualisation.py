import numpy as np
from util import load_hdf5, save_hdf5, load_json, save_json


def load_dataset():
    version_suffix = ''
    subset = 'kptest'
    meta_file = 'data/%svqa_std_mscoco_%s.meta' % (version_suffix, subset)
    data_file = 'data/%svqa_std_mscoco_%s.data' % (version_suffix, subset)
    ans_file = 'data/%sanswer_std_mscoco_%s.data' % (version_suffix, subset)
    # load meta file
    d = load_json(meta_file)
    images = d['images']
    quest_ids = d['quest_id']
    # load data file
    d = load_hdf5(data_file)
    quest = d['quest_arr'].astype(np.int32)
    quest_len = d['quest_len'].astype(np.int32)
    answer = d['answer'].astype(np.int32)
    # load answer file
    d = load_hdf5(ans_file)
    ans_tokens = d['ans_arr']
    ans_len = d['ans_len']
    return [images, quest_ids, quest, quest_len, answer, ans_tokens, ans_len]


def rand_perm(num):
    order = np.arange(num)
    np.random.shuffle(order)
    return order


def slice_dataset(inputs, num_keep=600):
    images, quest_ids, quest, quest_len, answer, ans_tokens, ans_len = inputs
    num = len(images)
    order = rand_perm(num)
    # random shuffle
    images = [images[_idx] for _idx in order]
    quest_ids = [quest_ids[_idx] for _idx in order]
    quest = quest[order]
    quest_len = quest_len[order]
    answer = answer[order]
    ans_tokens = ans_tokens[order]
    ans_len = ans_len[order]

    # prepare to run selection
    new_images, new_qids, new_q, new_q_len, new_a, new_aarr, new_alen = [], [], [], [], [], [], []
    num_selected = 0
    in_batch_image_key = {}
    for i in range(num):
        a = answer[i]
        image_key = images[i]
        if a == 2000:
            continue
        if image_key in in_batch_image_key:
            continue
        # add image to key
        in_batch_image_key[image_key] = None
        # add sample
        new_images.append(images[i])
        new_qids.append(quest_ids[i])
        new_q.append(quest[i])
        new_q_len.append(quest_len[i])
        new_a.append(answer[i])
        new_aarr.append(ans_tokens[i])
        new_alen.append(ans_len[i])
        num_selected += 1
        if num_selected == num_keep:
            break

    # expand for concat
    new_q = [_x[np.newaxis, :] for _x in new_q]
    new_aarr = [_x[np.newaxis, :] for _x in new_aarr]

    # merge data
    new_q = np.concatenate(new_q).astype(np.int32)
    new_q_len = np.array(new_q_len).astype(np.int32)
    new_a = np.array(new_a).astype(np.int32)
    new_aarr = np.concatenate(new_aarr).astype(np.int32)
    new_alen = np.array(new_alen).astype(np.int32)
    import pdb
    pdb.set_trace()

    # save
    version_suffix = ''
    subset = 'bs_test'
    meta_file = 'data/%svqa_std_mscoco_%s.meta' % (version_suffix, subset)
    data_file = 'data/%svqa_std_mscoco_%s.data' % (version_suffix, subset)
    ans_file = 'data/%sanswer_std_mscoco_%s.data' % (version_suffix, subset)
    save_json(meta_file, {'images': new_images,
                          'quest_id': new_qids})
    save_hdf5(data_file, {'quest_arr': new_q,
                          'quest_len': new_q_len,
                          'answer': new_a})
    save_hdf5(ans_file, {'ans_arr': new_aarr,
                         'ans_len': new_alen})


if __name__ == '__main__':
    dataset = load_dataset()
    slice_dataset(dataset, 600)


