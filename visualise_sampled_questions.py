from util import load_json, save_json
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
    writer = ExperimentWriter('latex/examples_replay_buffer_low_att')
    d = load_json('vqa_replay_buffer/low_att/vqa_replay.json')
    memory = d['memory']
    # show random 100

    if os.path.exists('vqa_replay_buffer/tmp_keys.json'):
        print('Loading keys')
        vis_keys = load_json('vqa_replay_buffer/tmp_keys.json')['keys']
    else:
        keys = deepcopy(memory.keys())
        np.random.seed(123)
        np.random.shuffle(keys)
        vis_keys = keys[:100]
        save_json('vqa_replay_buffer/tmp_keys.json', {'keys': vis_keys})

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
            conf = pathes[p]
            _tokens = [int(t) for t in p.split(' ')]
            sentence = to_sentence.index_to_question(_tokens)
            descr = '%s (%0.2f)' % (sentence, conf)
            questions.append(descr)
        writer.add_result(image_id, quest_id, im_path, head, questions)
    writer.render()


if __name__ == '__main__':
    visualise()
