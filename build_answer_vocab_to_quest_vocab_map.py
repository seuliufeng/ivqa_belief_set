from collections import OrderedDict


def load_vocab(fpath, reverse=False):
    vocab = []
    with open(fpath, 'r') as fs:
        for line in fs:
            word = line.split(' ')[0]
            vocab.append(word)
    if reverse:
        return vocab
    vocab_dict = OrderedDict()
    for i, word in enumerate(vocab):
        vocab_dict[word] = i
    return vocab_dict


def append_vocabulary(quest_vocab, reverse_ans_vocab):
    import numpy as np
    q_voc_ptr = len(quest_vocab)
    mapping = np.zeros(len(reverse_ans_vocab),
                       dtype=np.int32)  # dict size
    import pdb
    for a_id, word in enumerate(reverse_ans_vocab):
        if word in quest_vocab:
            mapping[a_id] = quest_vocab[word]
        else:
            quest_vocab[word] = q_voc_ptr
            mapping[a_id] = q_voc_ptr
            q_voc_ptr += 1
    qinds = quest_vocab.values()
    num = len(quest_vocab)
    pdb.set_trace()
    # assert(max(qinds) == num)
    assert (np.all(np.sort(qinds) == np.arange(num)))
    return quest_vocab, mapping


def dump_dict_and_mapping(quest_vocab, mapping):
    reverse_qvoc = {v: k for (k, v) in quest_vocab.iteritems()}
    with open('data/vqa_trainval_merged_word_counts', 'w') as fs:
        for i in range(len(reverse_qvoc)):
            fs.write('%s %d\n' % (reverse_qvoc[i], i))
    from util import save_hdf5
    save_hdf5('data/answer_index_to_merged_index.mapping',
              {'mapping': mapping})


def process():
    quest_voc_file = 'data/vqa_trainval_question_word_counts.txt'
    ans_voc_file = 'data/vqa_trainval_answer_word_counts.txt'

    quest_vocab = load_vocab(quest_voc_file)
    reverse_ans_vocab = load_vocab(ans_voc_file, reverse=True)

    def _in_vocab_stat(n_max):
        count = 0
        for i in range(n_max):
            word = reverse_ans_vocab[i]
            if word in quest_vocab:
                count += 1
        print('%d top %d words in question vocabulary' % (count, n_max))

    _in_vocab_stat(5000)
    _in_vocab_stat(8000)

    quest_vocab, mapping = append_vocabulary(quest_vocab, reverse_ans_vocab)
    dump_dict_and_mapping(quest_vocab, mapping)


if __name__ == '__main__':
    process()
