import tensorflow as tf


def _load_vocab(vocab_file):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    reverse_vocab = [line.split()[0] for line in reverse_vocab]
    return {word: id for id, word in enumerate(reverse_vocab)}



def compare_dicts(dict1, dict2):
    vocab_1 = _load_vocab(dict1)
    vocab_2 = _load_vocab(dict2)
    v1_size = len(vocab_1)
    v2_size = len(vocab_2)
    # compute intersect
    num_inter = 0
    for k in vocab_1.keys():
        if k in vocab_2:
            num_inter += 1

    # compute union
    vocab_1.update(vocab_2)
    num_union = len(vocab_1)
    print('vocab1_size: %d' % v1_size)
    print('vocab2_size: %d' % v2_size)
    print('union_size: %d' % num_union)
    print('intersect_size: %d' % num_inter)


if __name__ == '__main__':
    d1 = 'data/vqa_trainval_question_word_counts.txt'
    d2 = 'data2/v7w_train_question_word_counts.txt'
    compare_dicts(d1, d2)

    d1 = 'data/vqa_trainval_answer_word_counts.txt'
    d2 = 'data2/v7w_train_answer_word_counts.txt'
    compare_dicts(d1, d2)