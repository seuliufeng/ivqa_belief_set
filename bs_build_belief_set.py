from time import time
import os
from util import load_json, save_json
from bs_score_final_candidates_mlb_vqa2 import AttentionModel as MLB2AttModel
try:
    from n2mn_wrapper import N2MNWrapper as N2NMNModel
except:
    N2NMNModel = None
from vqa_interactive_ui import VanillaModel
from vqa_interactive_ui import AttentionModel as MLBAttModel
from w2v_answer_encoder import MultiChoiceQuestionManger

_TYPE2Model = {'MLB2-att': MLB2AttModel,
               'Vanilla': VanillaModel,
               'MLB-att': MLBAttModel,
               'N2NMN': N2NMNModel}


def compare_answer(a1, a2):
    return a1.lower().strip() == a2.lower().strip()


def load_lm_outputs(method, inf_type='rand'):
    assert (inf_type in ['beam', 'rand'])
    if inf_type == 'rand':
        res_file = 'result/bs_RL2_cands_LM_%s.json' % method
    else:
        res_file = 'result/bs_RL2_cands_LM_%s_BEAM.json' % method
    return load_json(res_file)


def process(method, inf_type='rand'):
    if inf_type == 'rand':
        res_file = 'result/bs_RL2_final_%s.json' % method
    else:
        res_file = 'result/bs_RL2_final_%s_BEAM.json' % method
    if os.path.exists(res_file):
        print('File %s already exist, skipped' % res_file)
        return

    # cands = load_results()
    model = _TYPE2Model[method]()
    mc_ctx = MultiChoiceQuestionManger(subset='val')

    task_data = load_lm_outputs(method, inf_type)

    belief_sets = []
    t = time()
    num = len(task_data)
    for i, quest_id_key in enumerate(task_data.keys()):
        # time it
        avg_time = (time() - t)
        print('%d/%d (%0.2f sec/sample)' % (i, num, avg_time))
        t = time()

        # extract basis info
        quest_id = int(quest_id_key)
        gt_answer = mc_ctx.get_gt_answer(quest_id)
        image_id = mc_ctx.get_image_id(quest_id)
        image = mc_ctx.get_image_file(quest_id)

        # process
        cands = task_data[quest_id_key]
        gt_question = mc_ctx.get_question(quest_id)

        i_scores, i_questions = [], []
        for item in cands:
            target = item['question']
            pred_ans, vqa_score = model.get_score(image_id, target)
            # inset check
            is_valid = compare_answer(pred_ans, gt_answer)
            if not is_valid:
                continue
            i_questions.append(target)
            i_scores.append([float(vqa_score), item['score']])
        print('%d/%d' % (len(i_questions), len(cands)))
        bs_i = {'image': image,
                'image_id': image_id,
                'question': gt_question,
                'answer': gt_answer,
                'belief_sets': i_questions,
                'belief_strength': i_scores}

        belief_sets.append(bs_i)
    save_json(res_file, belief_sets)


if __name__ == '__main__':
    # models = ['MLB2-att', 'Vanilla', 'MLB-att']
    models = ['N2NMN']
    for model in models:
        process(model, 'beam')
