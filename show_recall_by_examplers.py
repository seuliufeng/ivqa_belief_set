from extract_vqa_word2vec_coding import load_and_process_metadata
from extract_vqa_word2vec_coding import SentenceEncoder


def build_answer_vocab(meta):
    ans_vocab = {}
    for info in meta:
        _ans = info.answer
        if _ans in ans_vocab:
            ans_vocab[_ans] += 1
        else:
            ans_vocab[_ans] = 1
    return ans_vocab


def filter_answer_vocab(vocab, nkeep=5000):
    print('\nFilter: %d' % nkeep)
    counts = vocab.values()
    import numpy as np
    thresh = np.sort(counts)[-nkeep]
    new_dict = {}
    for key in vocab:
        if vocab[key] > thresh:
            new_dict[key] = vocab[key]
    return new_dict


def filter_answer_vocab_by_count(vocab, thresh=1):
    print('\nFilter: thresh %d' % thresh)
    new_dict = {}
    for key in vocab:
        if vocab[key] > thresh:
            new_dict[key] = vocab[key]
    return new_dict


def dict_intersect(train_voc, val_voc):
    d_inter_train = {}
    d_inter_val = {}
    for key in val_voc.keys():
        if key in train_voc:
            d_inter_val[key] = val_voc[key]
            d_inter_train[key] = train_voc[key]
    # print
    print('Number of unique answers in train: %d' % len(train_voc))
    print('Number of unique answers in val: %d' % len(val_voc))
    print('Number of shared answers (inter): %d' % len(d_inter_val))
    tot_cover = sum(d_inter_train.values()) / float(sum(train_voc.values())) * 100.
    print('Percentage of train QAs can be answered by inter: %0.2f' % tot_cover)
    tot_cover = sum(d_inter_val.values()) / float(sum(val_voc.values())) * 100.
    print('Percentage of val QAs can be answered by inter: %0.2f' % tot_cover)


def dict_intersect_with_ref(train_voc, val_voc, ref_voc):
    d_inter_train = {}
    d_inter_val = {}
    for key in val_voc.keys():
        if key in train_voc:
            d_inter_val[key] = val_voc[key]
            d_inter_train[key] = train_voc[key]
    # print
    print('Number of unique answers in train: %d' % len(train_voc))
    print('Number of unique answers in val: %d' % len(val_voc))
    print('Number of shared answers (inter): %d' % len(d_inter_val))
    tot_cover = sum(train_voc.values()) / float(sum(ref_voc.values())) * 100.
    print('Percentage of train QAs can be answered by inter: %0.2f' % tot_cover)
    tot_cover = sum(d_inter_val.values()) / float(sum(val_voc.values())) * 100.
    print('Percentage of val QAs can be answered by inter: %0.2f' % tot_cover)


def sort_dict_by_values(voc):
    from collections import OrderedDict
    sorted_voc = OrderedDict(sorted(voc.items(), key=lambda (k, v): -v))
    return sorted_voc


def encode_top_answers(voc):
    import numpy as np
    from scipy.spatial.distance import cdist
    encoder = SentenceEncoder()
    encoding = []
    index2ans = []
    print('Extracting answer codings')
    for i, ans in enumerate(voc):
        w2v = encoder.encode(ans)
        encoding.append(w2v)
        key = '_'.join(v.strip() for v in ans.split(','))
        key = '%s:%d(%d)' % (key, voc[ans], i+1)
        index2ans.append(key)
    ans_enc = np.concatenate(encoding)
    # l2 norm
    print('Normalise and compute distance')
    _norm = np.sqrt(np.square(ans_enc).sum(axis=1)) + 1e-8
    ans_enc /= _norm[:, np.newaxis]
    num = ans_enc.shape[0]
    dist = cdist(ans_enc, ans_enc)
    dist += np.eye(num) * dist.max()
    inds = dist.argsort(axis=1)

    K = 10
    print('Writing logs')
    with open('answer_nn.txt', 'w') as fs:
        for i in range(num):
            fs.write('Answer %d: %s\n' % (i + 1, index2ans[i]))
            nn_str = ', '.join([index2ans[_idx] for _idx in inds[i, :K]])
            fs.write('NNs: %s\n\n' % nn_str)


def main():
    train = load_and_process_metadata('train')
    val = load_and_process_metadata('val')
    # build answer vocab
    train_voc = build_answer_vocab(train)
    val_voc = build_answer_vocab(val)
    #
    # dict_intersect(train_voc, val_voc)
    # filter 5000
    # ft_voc = filter_answer_vocab(train_voc, nkeep=5000)
    # dict_intersect_with_ref(ft_voc, val_voc, train_voc)
    # filter 3000
    # ft_voc = filter_answer_vocab(train_voc, nkeep=3000)
    # dict_intersect_with_ref(ft_voc, val_voc, train_voc)
    # filter 2000
    # ft_voc = filter_answer_vocab(train_voc, nkeep=2000)
    # dict_intersect_with_ref(ft_voc, val_voc, train_voc)
    #
    # filter by counts
    # for t in range(5):
    t = 1
    ft_voc = filter_answer_vocab_by_count(train_voc, t)
    dict_intersect_with_ref(ft_voc, val_voc, train_voc)
    ft_voc = sort_dict_by_values(ft_voc)
    #
    encode_top_answers(ft_voc)


if __name__ == '__main__':
    main()
