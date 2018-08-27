from readers.vqa_naive_cst_data_fetcher import ContrastiveDataReader
from post_process_variation_questions import _parse_gt_questions
from inference_utils.question_generator_util import SentenceGenerator
import numpy as np


def test_cst_reader():
    reader = ContrastiveDataReader(batch_size=4)
    to_sentence = SentenceGenerator(trainset='trainval')

    reader.start()
    for i in range(4):
        images, quest, quest_len, top_ans, mask = reader.pop_batch()
        questions = _parse_gt_questions(quest, quest_len)
        print('\nBatch %d' % i)
        this_batch_size = images.shape[0] / 2
        for idx in range(this_batch_size):
            print('Real: %s' % to_sentence.index_to_question(questions[idx]))
            print('Fake: %s\n' % to_sentence.index_to_question(questions[idx + this_batch_size]))
        print('Mask:')
        print(mask.astype(np.int32))
    reader.stop()


if __name__ == '__main__':
    test_cst_reader()

