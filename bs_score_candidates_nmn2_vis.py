from time import time
from util import load_json, save_json
from n2mn_wrapper import N2MNWrapper
from w2v_answer_encoder import MultiChoiceQuestionManger


def load_results():
    return load_json('result/bs_cand_for_vis.json')


def compare_answer(a1, a2):
    return a1.lower().strip() == a2.lower().strip()


def process():
    cands = load_results()
    model = N2MNWrapper()
    mc_ctx = MultiChoiceQuestionManger(subset='val')

    results = []
    t = time()
    for i, res_i in enumerate(cands):
        if i % 100 == 0:
            avg_time = (time() - t) / 100.
            print('%d/%d (%0.2f sec/sample)' % (i, len(cands), avg_time))
            t = time()

        image_id = res_i['image_id']
        aug_id = res_i['aug_id']
        question = res_i['target']
        # question_id = int(aug_id / 1000)
        question_id = res_i['question_id']
        gt_answer = mc_ctx.get_gt_answer(question_id)
        pred_answers, scores = model.inference(image_id, [question])
        sc = scores[0]
        pred_ans = pred_answers[0]
        is_valid = compare_answer(pred_ans, gt_answer)
        # import pdb
        # pdb.set_trace()
        if not is_valid:
            continue
        t_i = {'image_id': int(image_id),
               'aug_id': aug_id,
               'question_id': question_id,
               'question': question,
               'score': float(sc)}
        results.append(t_i)
    save_json('result/bs_vis_scores_n2mn.json', results)


if __name__ == '__main__':
    process()
