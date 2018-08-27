import numpy as np
from util import unpickle
from inference_utils import vocabulary
from collections import OrderedDict

# -----------------------------------------------------------------------------#
# Specify model and table locations here
# -----------------------------------------------------------------------------#
path_to_models = '/home/fl302/Projects/tf_projects/text-to-image/Data/skipthoughts/'
path_to_tables = '/home/fl302/Projects/tf_projects/text-to-image/Data/skipthoughts/'
# -----------------------------------------------------------------------------#

path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'


def load_word_embedding():
    words = []
    utable = np.load(path_to_tables + 'utable.npy')
    f = open(path_to_tables + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    return utable


def reduce_word_embedding():
    embed_dim = 620
    target_dict = 'data/vqa_trainval_question_answer_word_counts.txt'
    utable = load_word_embedding()
    vocab = vocabulary.Vocabulary(target_dict).reverse_vocab
    vocab_size = len(vocab)
    embedding = np.zeros([vocab_size + 1, embed_dim], dtype=np.float32)
    missed = 0
    for i, word in enumerate(vocab):
        if word == '</S>':
            word = '<eos>'
        try:
            embedding[i, :] = utable[word]
        except Exception, e:
            missed += 1
            embedding[i, :] = utable['UNK']
    embedding[-1, :] = utable['UNK']  # add UNK
    print('Copied %d entries, missed %d entries' % (vocab_size - missed, missed))
    return embedding


def convert_weight_matrix(pretrain_file, tf_modelfile):
    print('Loading word embedding...')
    embedding = reduce_word_embedding()
    print('Loading models...')
    d = unpickle(pretrain_file)
    vars = {}
    print('Converting model...')
    for vname in d.keys():
        vars[vname] = d[vname].get_value()
    # slice theano variables
    gWx = vars['encoder_W']  # 620x4800
    gWh = vars['encoder_U']  # 2400x4800
    gb = vars['encoder_b']   # 1x4800
    U = vars['encoder_Ux']   # 2400x2400
    W = vars['encoder_Wx']   # 620x2400
    b = vars['encoder_bx']   # 1x2400
    # arrange to tensorflow format
    gate_w = np.concatenate([gWx, gWh], axis=0)
    gate_b = gb
    cand_w = np.concatenate([W, U], axis=0)
    cand_b = b
    weight_dict = {'Gates/Linear/Matrix': gate_w,
                   'Gates/Linear/Bias': gate_b,
                   'Candidate/Linear/Matrix': cand_w,
                   'Candidate/Linear/Bias': cand_b,
                   'word_embedding/map': embedding}
    print('Saving...')
    np.save(tf_modelfile, weight_dict)
    print('Model converted successfully!')


def convert_weight_matrix_default(pretrain_file, tf_modelfile):
    print('Loading word embedding...')
    embedding = reduce_word_embedding()
    print('Loading models...')
    d = unpickle(pretrain_file)
    vars = {}
    print('Converting model...')
    for vname in d.keys():
        vars[vname] = d[vname].get_value()
    # slice theano variables
    gWx = vars['encoder_W']  # 620x4800
    gWh = vars['encoder_U']  # 2400x4800
    gb = vars['encoder_b']   # 1x4800
    U = vars['encoder_Ux']   # 2400x2400
    W = vars['encoder_Wx']   # 620x2400
    b = vars['encoder_bx']   # 1x2400
    # arrange to tensorflow format
    gate_w = np.concatenate([gWx, gWh], axis=0)
    gate_b = gb
    candx_w = W
    candu_w = U
    candx_b = b
    weight_dict = {'Gates/Linear/Matrix': gate_w,
                   'Gates/Linear/Bias': gate_b,
                   'CandidateW/Linear/Matrix': candx_w,
                   'CandidateW/Linear/Bias': candx_b,
                   'CandidateU/Linear/Matrix': candu_w,
                   'word_embedding/map': embedding}
    print('Saving...')
    np.save(tf_modelfile, weight_dict)
    print('Model converted successfully!')


if __name__ == '__main__':
    convert_weight_matrix_default('../tf_projects/text-to-image/pretain_umodel.pkl',
                          '/home/fl302/Projects/tf_projects/text-to-image/skip_thought_default_qa.npy')
