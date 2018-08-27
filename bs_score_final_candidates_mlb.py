from time import time
from util import load_json, save_json
# from n2mn_wrapper import N2MNWrapper
from vqa_interactive_ui import AttentionModel, VanillaModel
from w2v_answer_encoder import MultiChoiceQuestionManger


def load_results():
    return load_json('result/samples_to_score.json')


def compare_answer(a1, a2):
    return a1.lower().strip() == a2.lower().strip()


def process(model_type='mlb'):
    cands = load_results()
    if model_type == 'mlb':
        model = AttentionModel()
    else:
        model = VanillaModel()
    mc_ctx = MultiChoiceQuestionManger(subset='val')

    results = {}
    t = time()
    for i, res_key in enumerate(cands):
        if i % 100 == 0:
            avg_time = (time() - t) / 100.
            print('%d/%d (%0.2f sec/sample)' % (i, len(cands), avg_time))
            t = time()
        res_i = cands[res_key]
        image_id = res_i['image_id']
        question = res_i['target']
        question_id = res_i['question_id']
        gt_answer = mc_ctx.get_gt_answer(question_id)
        pred_ans, scores = model.get_score(image_id, question)
        sc = float(scores)
        is_valid = compare_answer(pred_ans, gt_answer)
        # if not is_valid:
        #     continue
        results[res_key] = {'pred_answer': pred_ans,
                            'pred_score': sc,
                            'gt_answer': gt_answer,
                            'is_valid': is_valid}
    save_json('result/%s_scores_final_v2.json' % model_type, results)


if __name__ == '__main__':
    process()
