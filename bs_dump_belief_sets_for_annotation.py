import os
import numpy as np
# from util import load_json, save_json
import json
from copy import deepcopy


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


def belief_set_iou(results1, results2):
    recalls = []
    for res1, res2 in zip(results1, results2):
        # build key from res2
        dst_d = {q: None for q in res2['belief_sets']}
        num_in = 0
        assert (res1['image_id'] == res2['image_id'])
        assert (res1['question'] == res2['question'])
        for qry in res1['belief_sets']:
            if qry in dst_d:
                num_in += 1
        num_tot = len(res1['belief_sets'])
        EPS = 1e-8
        recall_i = float(num_in) / float(num_tot + EPS)
        if num_tot == 0:
            recall_i = 1.0
        recalls.append(recall_i)
    return 100. * np.array(recalls, dtype=np.float32).mean()
    # res1 = results1[res_k1]
    # res2 = results1[res_k2]


def _merge_one(s1, s2):
    # build dict
    d1 = {q: sc for q, sc in zip(s1['belief_sets'], s1['belief_strength'])}
    d2 = {q: sc for q, sc in zip(s2['belief_sets'], s2['belief_strength'])}
    assert (s1['image_id'] == s2['image_id'])
    assert (s1['question'] == s2['question'])
    d1.update(d2)
    ms = deepcopy(s1)
    bs, scores = [], []
    for k, v in d1.items():
        bs.append(k)
        scores.append(v)
    ms['belief_sets'] = bs
    ms['belief_strength'] = scores
    return ms


def merge_belief_sets(bs1, bs2):
    belief_sets = []
    for s1, s2 in zip(bs1, bs2):
        ms = _merge_one(s1, s2)
        belief_sets.append(ms)
    return belief_sets


def stat_nonzero_entry(results):
    idx = 0
    counts = []
    for res in results:
        bs = res['belief_sets']
        counts.append(len(bs))
        if len(bs) == 0:
            continue
        idx += 1
    avg_count = np.array(counts, dtype=np.float32).mean()
    print('Entries: %d/%d (avg. count: %0.2f)' % (idx, len(results), avg_count))


def check_stat():
    models = ['MLB2-att', 'Vanilla', 'MLB-att', 'N2NMN']
    sample_methods = ['beam', 'rand']
    # sample_methods = ['beam']
    train_methods = ['RL2', 'SL']

    sample_methods = ['rand', 'beam']
    train_methods = ['RL2', 'SL']
    for model in models:
        print('Method: %s' % model)
        for train_method in train_methods:
            bf_sets = []
            for sample_method in sample_methods:
                bs = load_belief_sets(model, sample_method, train_method)
                # stat_nonzero_entry(bs)
                bf_sets.append(bs)
            # compute iou
            iou = belief_set_iou(*bf_sets)
            print('%s\t: %0.2f % s' % (train_method, iou, ' in '.join(sample_methods)))

    sample_methods = ['beam', 'rand']
    train_methods = ['SL', 'RL2']
    for model in models:
        print('Method: %s' % model)
        for sample_method in sample_methods:
            bf_sets = []
            for train_method in train_methods:
                bs = load_belief_sets(model, sample_method, train_method)
                # stat_nonzero_entry(bs)
                bf_sets.append(bs)
            # compute iou
            iou = belief_set_iou(*bf_sets)
            print('%s\t: %0.2f % s' % (sample_method, iou, ' in '.join(train_methods)))


def make_blacklist_key(image, question):
    key = '#%s#%s' % (os.path.basename(image), question)
    return key


def sample_belief_set(bs, method, nsample=350):
    is_valid = [len(s['belief_sets']) > 0 for s in bs]
    valid_ids = np.where(is_valid)[0]
    np.random.shuffle(valid_ids)
    sample_inds = valid_ids[:nsample]
    black_list = {}
    task_data, task_data_state = [], []
    for sidx in sample_inds:
        s = bs[sidx]
        bsets = s['belief_sets']
        bsores = s['belief_strength']
        idx = np.random.randint(len(bsets))
        target = bsets[idx]
        vqa_score = bsores[idx][0]
        answer = s['answer']
        question = s['question']
        image = os.path.basename(s['image'])
        aug_id = idx
        task_data.append({'image': image,
                          'aug_id': aug_id,
                          'question': question,
                          'answer': answer,
                          'target': target,
                          'score': vqa_score})
        task_data_state.append(1)
        black_list[make_blacklist_key(image, question)] = target
    # save results
    sv_dir = 'works/%s' % method
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    save_json(os.path.join(sv_dir, 'task_data.json'), task_data)
    save_json(os.path.join(sv_dir, 'task_data_state.json'), task_data_state)
    return black_list


def apply_black_list(bs, black_list):
    for item in bs:
        image = os.path.basename(item['image'])
        question = item['question']
        bkey = make_blacklist_key(image, question)
        num = len(item['belief_sets'])
        if bkey in black_list:
            cands = item['belief_sets']
            scores = item['belief_strength']
            hash_t = {c: i for i, c in enumerate(cands)}
            query = black_list[bkey]
            if query in hash_t:
                pre_del_d = {c: s[0] for c, s in zip(cands, scores)}
                print('Deleting element from %s' % bkey)
                idx = hash_t[query]
                # remove blacklist elements
                del item['belief_sets'][idx]
                del item['belief_strength'][idx]
                # check correctness
                for c, s in zip(item['belief_sets'], item['belief_strength']):
                    assert (pre_del_d[c] == s[0])
            # print('%d-%d' % (len(item['belief_sets']), num))
    return bs


def sample_belief_set_round2(bs, method, black_list, nsample=350):
    bs = apply_black_list(bs, black_list)
    is_valid = [len(s['belief_sets']) > 0 for s in bs]
    valid_ids = np.where(is_valid)[0]
    np.random.shuffle(valid_ids)
    sample_inds = valid_ids[:nsample]
    task_data, task_data_state = [], []
    for sidx in sample_inds:
        s = bs[sidx]
        bsets = s['belief_sets']
        bsores = s['belief_strength']
        idx = np.random.randint(len(bsets))
        target = bsets[idx]
        vqa_score = bsores[idx][0]
        answer = s['answer']
        question = s['question']
        image = os.path.basename(s['image'])
        aug_id = idx
        task_data.append({'image': image,
                          'aug_id': aug_id,
                          'question': question,
                          'answer': answer,
                          'target': target,
                          'score': vqa_score})
        task_data_state.append(1)
    # save results
    sv_dir = 'works/%s' % method
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    save_json(os.path.join(sv_dir, 'task_data.json'), task_data)
    save_json(os.path.join(sv_dir, 'task_data_state.json'), task_data_state)


def main():
    models = ['MLB2-att', 'Vanilla', 'MLB-att', 'N2NMN']
    # models = ['MLB-att']
    # sample_methods = ['beam', 'rand']
    sample_methods = ['beam']
    train_methods = ['RL2', 'SL']
    # train_methods = ['RL2']

    for model in models:
        print('Method: %s' % model)
        merged = []
        for train_method in train_methods:
            bf_sets = []
            for sample_method in sample_methods:
                bs = load_belief_sets(model, sample_method, train_method)
                # stat_nonzero_entry(bs)
                bf_sets.append(bs)
            # mbs = merge_belief_sets(*bf_sets)
            mbs = bf_sets[0]
            merged.append(mbs)
        # sample
        blacklist = sample_belief_set(merged[0], '%s_%s' % (train_methods[0], model))
        sample_belief_set_round2(merged[1], '%s_%s' % (train_methods[1], model), blacklist)


if __name__ == '__main__':
    main()
