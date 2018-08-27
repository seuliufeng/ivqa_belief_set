from extract_vqa_word2vec_coding import SentenceEncoder
import numpy as np
from scipy.spatial.distance import cdist


def extract_w2v():
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    answer_vocab = []
    with open(top_ans_file, 'r') as fs:
        for line in fs:
            answer_vocab.append(line.strip())
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
    _norm = np.sqrt(np.square(ans_enc).sum(axis=1)) + 1e-8
    ans_enc /= _norm[:, np.newaxis]
    num = ans_enc.shape[0]
    # dist = cdist(ans_enc, ans_enc)
    sim = np.dot(ans_enc, ans_enc.transpose())
    from util import save_hdf5
    save_hdf5('data/top2000_answer_feat.data', {'ans_w2v': ans_enc,
                                                'sim': sim})


if __name__ == '__main__':
    extract_w2v()
