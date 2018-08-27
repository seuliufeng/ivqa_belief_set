import tensorflow as tf
from util import load_hdf5


class PinnedDataManager(object):
    def __init__(self, data):
        _data = tf.constant(data, dtype=tf.int32)
        data_shape = _data.get_shape().as_list()
        self._datum_shape = data_shape[1:]
        self._num_datum = data_shape[0]
        self._data = tf.reshape(_data, [self._num_datum, -1])

    def get_data_by_index(self, indices):
        batch_data = tf.gather(self._data, indices)
        # pre-precessing and data augmentation
        batch_data = tf.reshape(batch_data,
                                shape=[-1]+self._datum_shape)
        return batch_data


class TopAnswerDataLayer(object):
    def __init__(self, top_ans_file, k):
        print('Answer layer: Loading top answer sequences')
        d = load_hdf5(top_ans_file)
        self.k = k
        self.data_len = d['answer_seq'].shape[-1]
        self._answer_seq = PinnedDataManager(d['answer_seq'])
        self._answer_len = PinnedDataManager(d['answer_seq_len'])

    def get_top_answer_sequences(self, top_k_indices):
        k = tf.shape(top_k_indices)[1]
        ans_seq = self._answer_seq.get_data_by_index(top_k_indices)
        ans_len = self._answer_len.get_data_by_index(top_k_indices)
        ans_seq = tf.reshape(ans_seq, [-1, k, self.data_len])
        ans_len = tf.reshape(ans_len, [-1, k])
        return ans_seq, ans_len

    @property
    def max_len(self):
        return self.data_len


def load_top_answer_list():
    top_answer_vocab = 'data/vqa_trainval_top2000_answers.txt'
    with open(top_answer_vocab, 'r') as fs:
        lines = fs.readlines()
    top_answers = [l.strip() for l in lines]
    return top_answers


def make_top_answer_data_layer():
    from build_v2_ivqa_data import _load_vocab, _list_tokens_to_array
    from nltk.tokenize import word_tokenize
    from util import save_hdf5
    answer_vocab_file = 'data/vqa_trainval_answer_word_counts.txt'
    vocab = _load_vocab(answer_vocab_file)
    top_answers = load_top_answer_list()

    answers = []
    for top_ans in top_answers:
        tokenized = word_tokenize(str(top_ans).lower())
        token_ids = [vocab.word_to_id(word) for word in tokenized]
        answers.append(token_ids)
    ans_arr, ans_len = _list_tokens_to_array(answers)
    save_hdf5('data/top_answer2000_sequences.h5', {'answer_seq': ans_arr,
                                                   'answer_seq_len': ans_len})


def test_top_answer_layer():
    from inference_utils.question_generator_util import SentenceGenerator
    to_sentence = SentenceGenerator(trainset='trainval')

    def visualise_sequence(seqs, seqs_len, idx):
        seq = seqs[idx]
        seq_len = seqs_len[idx]
        vis_seq = seq[:seq_len]
        answer = to_sentence.index_to_answer(vis_seq)
        print('%s' % answer)
        return answer

    top_ans_file = 'data/top_answer2000_sequences.h5'
    answer_pool = TopAnswerDataLayer(top_ans_file, k=4)
    top_answer_list = load_top_answer_list()

    import numpy as np
    ind = np.random.randint(low=0, high=len(top_answer_list), size=[5, 4],
                            dtype=np.int32)
    top_k_ind = tf.constant(ind, dtype=tf.int32)
    t_ans_arr, t_ans_len = answer_pool.get_top_answer_sequences(top_k_ind)
    ans_arr = t_ans_arr.eval().reshape([-1, answer_pool.data_len])
    ans_len = t_ans_len.eval().reshape([-1])

    answer_ind = top_k_ind.eval().reshape([-1])

    num_test = ans_len.size
    num_passed = 0
    for i in range(num_test):
        top_ans = top_answer_list[answer_ind[i]]
        print(top_ans)
        seq_ans = visualise_sequence(ans_arr, ans_len, i)
        print('========================')
        num_passed += (seq_ans == top_ans)
    print('\nFinish test top answer layer\nPassed: %d/%d' % (num_passed, num_test))


if __name__ == '__main__':
    test_top_answer_layer()

