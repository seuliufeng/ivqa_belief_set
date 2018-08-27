from readers.ivqa_reader_creater import create_reader
from inference_utils.question_generator_util import SentenceGenerator


def main(_):
    batch_size = 4
    create_fn = create_reader('VAQ-2Att', phase='train')
    to_sentence = SentenceGenerator(trainset='trainval')

    def trim_sequence(seqs, seqs_len, idx):
        seq = seqs[idx]
        seq_len = seqs_len[idx]
        return seq[:seq_len]

    def test_reader(reader):
        reader.start()
        for i in range(5):
            inputs = reader.pop_batch()
            im, attr, capt, capt_len, ans_seq, ans_seq_len = inputs
            question = to_sentence.index_to_question(trim_sequence(capt, capt_len, 1))
            answer = to_sentence.index_to_answer(trim_sequence(ans_seq, ans_seq_len, 1))
            print('Q: %s\nA: %s\n' % (question, answer))
        reader.stop()
    print('v1:')
    reader = create_fn(batch_size, subset='kptrain', version='v1')
    test_reader(reader)
    del reader

    print('v2:')
    reader = create_fn(batch_size, subset='kptrain', version='v2')
    test_reader(reader)
    del reader

if __name__ == '__main__':
    main(None)
