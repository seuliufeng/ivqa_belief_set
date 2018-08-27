from collections import defaultdict
from util import load_json, save_json
from scipy.io import savemat
import numpy as np

_METHOD = 'union'
assert (_METHOD in ['union', 'max'])


def load_results(method):
    d = defaultdict()
    res_file = 'result/bs_gen_%s.json' % method
    results = load_json(res_file)
    print('num items: %d' % (len(results)))
    for item in results:
        # image_id = item['image_id']
        aug_quest_id = item['question_id']
        # question = item['question']
        # score = item['score']
        abs_quest_id = int(aug_quest_id / 1000)
        if abs_quest_id in d:
            d[abs_quest_id].append(item)
        else:
            d[abs_quest_id] = [item]
    print('average items: %0.2f' % (len(results)/float(len(d))))
    return d


def _merge_item(items1, items2):
    items = items1 + items2
    d = {}
    new_items = []
    for item in items:
        question = item['question']
        if question not in d:
            d[question] = None
            new_items.append(item)
        else:
            continue
    items = new_items
    scores = np.array([_t['score'] for _t in items])
    n1 = len(items1)
    n2 = len(items2)
    try:
        assert (n1 == n2)
    except:
        pass
        # print('Waring: not equal, n1: %d, n2: %d' % (n1, n2))
        # raise Exception('not equal')
    if _METHOD == 'union':
        n = n1 + n2
    elif _METHOD == 'max':
        n = n1
    else:
        raise Exception('unknown mode')
    pick_ids = np.argsort(-scores)[:n]
    abs_quest_id = int(items[0]['question_id'] / 1000)
    results = []
    for i, _pid in enumerate(pick_ids):
        item = items[_pid]
        question_id = abs_quest_id * 1000 + i
        item['question_id'] = question_id
        results.append(item)
    return results, scores[pick_ids], len(results)


def merge_result(res1, res2):
    results = []
    unk_counts = []
    batch_vqa_scores = []
    for quest_id in res1:
        items1 = res1[quest_id]
        items2 = res2[quest_id]
        res_i, scores_i, n_i = _merge_item(items1, items2)
        results += res_i
        batch_vqa_scores.append(scores_i.mean())
        unk_counts.append(n_i)
    # save results
    res_file = 'result/bs_gen_%s.json' % 'vae_ia_merge'
    score_file = 'result/bs_vqa_scores_%s.mat' % 'vae_ia_merge'
    save_json(res_file, results)
    batch_vqa_scores = np.array(batch_vqa_scores, dtype=np.float32)
    mean_vqa_score = batch_vqa_scores.mean()
    mean_unk_count = np.mean(unk_counts)

    savemat(score_file, {'scores': batch_vqa_scores, 'mean_score': mean_vqa_score})
    print('BS mean VQA score: %0.3f' % mean_vqa_score)
    print('BS mean #questions: %0.3f' % mean_unk_count)
    return res_file, mean_vqa_score, mean_unk_count


def main():
    res1 = load_results('vae_ia_rl_attention2_run0')
    # res1 = load_results('vae_ia_rl_attention2_run2')
    res2 = load_results('vae_rl1')
    # res2 = load_results('vae_ia_rl_mlb_r2')
    res_file, mean_vqa_score, mean_unk_count = merge_result(res1, res2)
    from eval_vqa_question_oracle import evaluate_oracle
    evaluate_oracle(res_file)
    print('BS mean VQA score: %0.3f' % mean_vqa_score)
    print('BS mean #questions: %0.3f' % mean_unk_count)


if __name__ == '__main__':
    main()
