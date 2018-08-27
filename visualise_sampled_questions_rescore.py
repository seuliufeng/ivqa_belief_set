from util import load_json
import os
from inference_utils.question_generator_util import SentenceGenerator
from write_examples import ExperimentWriter
from w2v_answer_encoder import MultiChoiceQuestionManger
from copy import deepcopy
import numpy as np

IM_ROOT = '/usr/data/fl302/data/VQA/Images/'


def visualise():
    mc_ctx = MultiChoiceQuestionManger()
    to_sentence = SentenceGenerator(trainset='trainval')
    # writer = ExperimentWriter('latex/examples_replay_buffer_rescore')
    writer = ExperimentWriter('latex/examples_replay_buffer_rescore_prior')
    # d = load_json('vqa_replay_buffer/vqa_replay_low_rescore.json')
    d = load_json('vqa_replay_buffer/vqa_replay_low_rescore_prior_05_04.json')
    memory = d['memory']
    # show random 100
    keys = deepcopy(memory.keys())
    np.random.seed(123)
    np.random.shuffle(keys)
    vis_keys = keys[:100]
    for i, quest_key in enumerate(vis_keys):
        pathes = memory[quest_key]
        if len(pathes) == 0:
            continue
        # if it has valid questions
        quest_id = int(quest_key)
        image_id = mc_ctx.get_image_id(quest_id)
        gt_question = mc_ctx.get_question(quest_id)
        answer = mc_ctx.get_gt_answer(quest_id)
        head = 'Q: %s A: %s' % (gt_question, answer)
        im_file = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        im_path = os.path.join(IM_ROOT, im_file)
        questions = []
        for p in pathes.keys():
            conf1, conf2 = pathes[p]
            _tokens = [int(t) for t in p.split(' ')]
            sentence = to_sentence.index_to_question(_tokens)
            descr = '%s (%0.2f-%0.2f)' % (sentence, conf1, conf2)
            questions.append(descr)
        writer.add_result(image_id, quest_id, im_path, head, questions)
    writer.render()


if __name__ == '__main__':
    visualise()
