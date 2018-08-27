from util import load_json, save_json
from time import time


def load_task_data():
    d = load_json('data/annotations/OpenEnded_mscoco_test-dev2015_questions.json')
    return d['questions']


def load_model(model_type):
    if model_type == 'MLB2-att':
        from bs_score_final_candidates_mlb_vqa2 import AttentionModel
        model = AttentionModel('test')
        return model
    elif model_type == 'Vanilla':
        from tmp_vanilla_vqa_model import VanillaModel
        model = VanillaModel()
        return model


def process(model_type='MLB2-att'):
    result_format = 'result/vqa_OpenEnded_mscoco_test-dev2015_%s_results.json'
    questions = load_task_data()
    model = load_model(model_type)
    results = []
    num = len(questions)
    t = time()
    for i, item in enumerate(questions):
        if i % 1000 == 0:
            print('Testing [%s]: %d/%d (speed: %0.2f sec/1000 samples)' % (model_type, i, num, time() - t))
            t = time()
        image_id = int(item['image_id'])
        question_id = int(item['question_id'])
        question = item['question']
        pred_ans, _ = model.get_score(image_id, question)
        results.append({u'answer': pred_ans, u'question_id': question_id})
    res_file = result_format % model_type
    save_json(res_file, results)


if __name__ == '__main__':
    # process()
    process('Vanilla')
