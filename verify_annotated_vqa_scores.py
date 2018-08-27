import os
from vqa_interactive_ui import AttentionModel
from util import load_json, save_json
from answer_token_to_top_answers import AnswerTokenToTopAnswer
import pdb


def process():
    def _parse_image_id(image):
        return int(image.split('.')[0].split('_')[-1])

    model = AttentionModel()
    ans2top_ans = AnswerTokenToTopAnswer()

    task_data_dir = '/usr/data/fl302/code/utils/bs_data_maker'
    task_data_file = os.path.join(task_data_dir, 'task_data_for_verif.json')
    task_data = load_json(task_data_file)
    is_valid = []
    num = len(task_data)
    for i, info in enumerate(task_data):
        print('%d/%d' % (i, num))
        image = info['image']
        image_id = _parse_image_id(image)
        question = info['target']
        answer = info['answer']
        scores = model.inference(image_id, question)
        scores[:, -1] = -10.
        # pdb.set_trace()
        top_ans_id = ans2top_ans.direct_query(answer)
        if top_ans_id == 2000:
            raise Exception('Warning: answer oov')
        scores = scores.flatten()
        pred_top_ans_id = scores.argmax()
        is_valid.append(int(pred_top_ans_id == top_ans_id))

    n_valid = sum(is_valid)
    print('valid: %d/%d' % (n_valid, num))
    save_json(os.path.join(task_data_dir, 'task_data_verif_state.json'), is_valid)


if __name__ == '__main__':
    process()




