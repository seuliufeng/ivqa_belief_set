import numpy as np
from util import load_json, save_json
from bs_language_model_wrapper import LanuageModelWrapper


def evaluate_question_standard(result_file, subset='kptest', version='v1'):
    from analysis.eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()

    subset = 'train' if subset == 'train' else 'val'
    if version == 'v1':
        annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
        question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)
    elif version == 'v2':
        anno_dir = '/import/vision-ephemeral/fl302/data/VQA2.0'
        annotation_file = '%s/v2_mscoco_%s2014_annotations.json' % (anno_dir, subset)
        question_file = '%s/v2_OpenEnded_mscoco_%s2014_questions.json' % (anno_dir, subset)
    else:
        raise Exception('unknown version, v1 or v2')

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results()
    # return evaluator.get_overall_blue4()
    return evaluator.get_overall_cider()



res_file = '/data1/fl302/projects/inverse_vqa/result/var_vaq_rand_IVQA-BASIC_full.json'
results = load_json(res_file)

lm = LanuageModelWrapper()

parsed_results = {}
parsed_questions = {}
for res in results:
    qid = int(res['question_id'] / 1000)
    sent = res['question_inds']
    if qid in parsed_results:
        parsed_results[qid].append(sent)
        parsed_questions[qid].append(res)
    else:
        parsed_results[qid] = [sent]
        parsed_questions[qid] = [res]

scored_qs = []
nums = []
scores = []
new_results = []
maxprob_results = []

max_prob_info = {}
for qid, qs in parsed_results.items():
    info = parsed_questions[qid]
    print('\nquestion_id: %d' % qid)
    scores_i = lm.inference(qs)
    scores_i = scores_i.tolist()

    # gscores = np.array([_i['probs'] for _i in info])
    gscores = np.array([_i['counts'] for _i in info])
    # gscores[np.array(scores_i) < 1 - 1e-4] = -float('inf')
    gscores[np.array(scores_i) < 1 - 1e-4] = -1
    max_index = gscores.argmax()
    maxprob_info = info[max_index]
    maxprob_results.append({'image_id': maxprob_info['image_id'],
                            'question_id': qid,
                            'question': maxprob_info['question']})

    for sc, _info in zip(scores_i, info):
        if sc > 0.999999:
            _aug_qid = _info['question_id']
            _image_id = _info['image_id']
            _sentence = _info['question']
            # print('%s (%0.2f)' % (_sentence, sc))
            res_i = {'image_id': int(_image_id),
                     'question_id': _aug_qid,
                     'question': _sentence}
            new_results.append(res_i)

    scores += scores_i
    nums.append(len(scores_i))
    scored_qs.append({'question_id': qid,
                      'question_inds': qs,
                      'lm_scores': scores_i})

maxprob_res_file = res_file = '/data1/fl302/projects/inverse_vqa/result/var_vaq_rand_IVQA-BASIC_lmflt_maxprob.json'
save_json(maxprob_res_file, maxprob_results)
evaluate_question_standard(maxprob_res_file)


sv_file = '/data1/fl302/projects/inverse_vqa/result/var_vaq_rand_IVQA-BASIC_lmscores.json'
save_json(sv_file, scored_qs)

print('Average #questions: %0.2f' % np.mean(nums))
print('Average LM scores: %0.2f' % np.mean(scores))

from eval_vqa_question_oracle import evaluate_oracle

new_res_file = '/data1/fl302/projects/inverse_vqa/result/var_vaq_rand_IVQA-BASIC_lmflt.json'
save_json(new_res_file, new_results)
cider = evaluate_oracle(new_res_file, split='val')

import pdb

pdb.set_trace()
