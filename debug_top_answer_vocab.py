from w2v_answer_encoder import MultiChoiceQuestionManger
from readers.ivqa_reader_creater import create_reader
from inference_utils.question_generator_util import SentenceGenerator
import numpy as np
import pdb


def test():
    top_ans_file = '/import/vision-ephemeral/fl302/code/' \
                   'VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    # top_ans_file = 'data/vqa_trainval_top2000_answers.txt'
    mc_ctx = MultiChoiceQuestionManger(subset='val', load_ans=True,
                                       top_ans_file=top_ans_file)
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)
    answer_enc = mc_ctx.encoder
    # quest_ids = mc_ctx._quest_id2image_id.keys()
    # quest_ids = np.array(quest_ids)

    # qids = np.random.choice(quest_ids, size=(5,), replace=False)

    create_fn = create_reader('VAQ-CA', 'train')
    reader = create_fn(batch_size=4, subset='kprestval')
    reader.start()

    for _ in range(20):
        # inputs = reader.get_test_batch()
        inputs = reader.pop_batch()

        _, _, _, _, labels, ans_seq, ans_len, quest_ids, image_ids = inputs

        b_top_ans = answer_enc.get_top_answers(labels)
        for i, (quest_id, i_a) in enumerate(zip(quest_ids, b_top_ans)):
            print('question id: %d' % quest_id)
            gt = mc_ctx.get_gt_answer(quest_id)
            print('GT: %s' % gt)
            print('Top: %s' % i_a)
            print('SG: top: %s' % to_sentence.index_to_top_answer(labels[i]))
            seq = ans_seq[i][:ans_len[i]].tolist()
            print('SG: seq: %s\n' % to_sentence.index_to_answer(seq))

    reader.stop()


if __name__ == '__main__':
    test()
