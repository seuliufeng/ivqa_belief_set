from util import load_json, save_json, find_image_id_from_fname
from time import time


def load_task_data():
    d = load_json('data/adv_hard_set.json')
    return d


def load_model(model_type):
    if model_type == 'MLB2-att':
        from bs_score_final_candidates_mlb_vqa2 import AttentionModel
        model = AttentionModel()
        return model
    elif model_type == 'Vanilla':
        from vqa_interactive_ui import VanillaModel
        model = VanillaModel()
        return model
    elif model_type == 'MLB-att':
        from vqa_interactive_ui import AttentionModel
        model = AttentionModel()
        return model
    elif model_type == 'N2NMN':
        from n2mn_wrapper import N2MNWrapper
        model = N2MNWrapper()
        return model
    elif model_type == 'MCB-att':
        from mcb_wrapper import MCBModel
        model = MCBModel()
        return model


def process(model_type='MLB2-att'):
    # result_format = 'result/vqa_pred_bsadv_%s_v2.json'
    result_format = 'result/vqa_pred_bsadv_%s.json'
    questions = load_task_data()
    model = load_model(model_type)
    results = []
    num = len(questions)
    t = time()
    acc = []
    for i, item in enumerate(questions):
        if i % 100 == 0:
            print('Testing [%s]: %d/%d (speed: %0.2f sec/100 samples)' % (model_type, i, num, time() - t))
            t = time()
        image_id = find_image_id_from_fname(item['image'])
        question = item['target']
        answer = item['answer']
        ans, _ = model.get_score(image_id, question)
        acc.append(ans == answer)
        # score = model.query_score(image_id, question, answer)
        results.append(ans)
    res_file = result_format % model_type
    save_json(res_file, results)
    print('%s: Acc: %0.2f' % (model_type, 100.*sum(acc)/float(len(acc))))


if __name__ == '__main__':
    # process()
    # algs = ['MLB-att', 'MLB2-att', 'Vanilla']
    # algs = ['N2NMN']
    algs = ['MCB-att']
    for alg in algs:
        process(alg)
