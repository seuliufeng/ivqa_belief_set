from readers.vqa_irrelevance_data_fetcher import AttentionDataReader as Reader
from post_process_variation_questions import _parse_gt_questions
from inference_utils.question_generator_util import SentenceGenerator

reader = Reader(batch_size=10,
                subset='trainval',
                model_name='something',
                epsilon=0.5,
                feat_type='res5c',
                version='v1',
                counter_sampling=False)

to_sentence = SentenceGenerator(trainset='trainval')

reader.start()

for i in range(5):
    print('--------- BATCH %d ---------' % i)
    res5c, quest, quest_len, labels = reader.pop_batch()
    pathes = _parse_gt_questions(quest, quest_len)
    for _p, lbl in zip(pathes, labels):
        print('%s %d' % (to_sentence.index_to_question(_p), lbl))

reader.stop()
