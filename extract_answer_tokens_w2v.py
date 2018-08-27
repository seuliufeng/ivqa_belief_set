from extract_vqa_word2vec_coding import SentenceEncoder
import numpy as np
from scipy.spatial.distance import cdist


def extract_w2v():
    trainset = 'trainval'
    top_ans_file = 'data/vqa_%s_answer_word_counts.txt' % trainset
    answer_vocab = []
    with open(top_ans_file, 'r') as fs:
        for line in fs:
            token = line.split(' ')[0].strip()
            answer_vocab.append(token)
    # extract w2v
    encoder = SentenceEncoder()
    encoding = []
    index2ans = []
    print('Extracting answer codings')

    for i, ans in enumerate(answer_vocab):
        w2v = encoder.encode(ans)
        encoding.append(w2v)
        key = '_'.join(v.strip() for v in ans.split(','))
        key = '%s:%d' % (key, i + 1)
        index2ans.append(key)

    ans_enc = np.concatenate(encoding)
    # l2 norm
    print('Normalise and compute distance')
    # _norm = np.sqrt(np.square(ans_enc).sum(axis=1)) + 1e-8
    # ans_enc /= _norm[:, np.newaxis]
    from util import save_hdf5
    save_hdf5('data/vqa_trainval_answer_vocab_w2v.data', {'ans_w2v': ans_enc})


if __name__ == '__main__':
    extract_w2v()
