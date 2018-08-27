import os
from util import load_json, save_json
from w2v_answer_encoder import MultiChoiceQuestionManger


def load_results(model_name):
    print('Load results of %s' % model_name)
    res_dir = '/homes/fl302/Desktop/annotation'
    fname = os.path.join(res_dir, '%s.json' % model_name)
    results = load_json(fname)

    d = {}
    for res in results:
        quest_id = res['question_id']
        if 'image_id' in res:
            image_id = res['image_id']
        else:
            image_id = None
        question = res['question']
        d[quest_id] = {'image_id': image_id,
                       'question_id': quest_id,
                       'question': [question]}
    return d


def init_saving_format(result, mc_ctx):
    for quest_id in result.keys():
        answer = mc_ctx.get_gt_answer(quest_id)
        result[quest_id]['answer'] = answer
    return result


def append_results(main, result):
    for quest_id in main.keys():
        main[quest_id]['question'] += result[quest_id]['question']
    return main


def dump_results(main, models):
    results = main.values()
    d = {'models': models, 'results': results}
    save_json('/homes/fl302/Desktop/annotation/combined_results.json', d)


def combine_models():
    models = ['a', 'i', 'i+at', 'nn', 'sat', 'vqg+vqa', 'ours']
    mc_ctx = MultiChoiceQuestionManger()

    d = {}
    for model in models:
        d[model] = load_results(model)

    res = {}
    for i, m in enumerate(models):
        print('Adding results of %s' % m)
        if i == 0:
            res = init_saving_format(d[m], mc_ctx)
        else:
            res = append_results(res, d[m])

    import pdb
    pdb.set_trace()
    dump_results(res, models)


if __name__ == '__main__':
    combine_models()