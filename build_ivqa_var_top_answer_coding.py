import tensorflow as tf
import numpy as np
from models.model_creater import get_model_creation_fn
from w2v_answer_encoder import MultiChoiceQuestionManger, _tokenize_sentence
from restorer import Restorer
from inference_utils.question_generator_util import SentenceGenerator


def put_to_array(sentences):
    sentence_lengths = [len(s) for s in sentences]
    max_length = max(sentence_lengths)
    batch_size = len(sentences)
    token_arrays = np.zeros([batch_size, max_length], dtype=np.int32)
    for s, s_len, target in zip(sentences, sentence_lengths, token_arrays):
        target[:s_len] = s
    token_lens = np.array(sentence_lengths, dtype=np.int32)
    return token_arrays.astype(np.int32), token_lens


def convert():
    model_name = 'ivaq_var_restval'
    checkpoint_path = 'model/var_ivqa_pretrain_restval/model.ckpt-505000'
    # build model
    from config import ModelConfig
    model_config = ModelConfig()
    model_fn = get_model_creation_fn('VAQ-Var')
    # create graph
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'beam')
        model.build()
        tf_embedding = model._answer_embed
        tf_answer_feed = model._ans
        tf_answer_len_feed = model._ans_len
        # Restore from checkpoint
        print('Restore from %s' % checkpoint_path)
        restorer = Restorer(g)
        sess = tf.Session()
        restorer.restore(sess, checkpoint_path)

    # build reader
    top_ans_file = '/import/vision-ephemeral/fl302/code/' \
                   'VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    mc_ctx = MultiChoiceQuestionManger(subset='val', load_ans=True,
                                       top_ans_file=top_ans_file)
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)
    answer_encoder = mc_ctx.encoder

    top_answer_inds = range(2000)
    top_answers = answer_encoder.get_top_answers(top_answer_inds)

    answer_seqs = answer_encoder.encode_to_sequence(top_answers)
    for i, (ans, seq) in enumerate(zip(top_answers, answer_seqs)):
        rec_ans = to_sentence.index_to_answer(seq)
        ans = ' '.join(_tokenize_sentence(ans))
        print('%d: Raw: %s, Rec: %s' % (i + 1, ans, rec_ans))
        assert (ans == rec_ans)
    print('Checking passed')

    # extract
    print('Converting...')
    ans_arr, ans_arr_len = put_to_array(answer_seqs)
    import pdb
    pdb.set_trace()
    embedding = sess.run(tf_embedding, feed_dict={tf_answer_feed: ans_arr.astype(np.int32),
                                                  tf_answer_len_feed: ans_arr_len.astype(np.int32)})
    # save
    sv_file = 'data/v1_%s_top2000_lstm_embedding.h5' % model_name
    from util import save_hdf5
    save_hdf5(sv_file, {'answer_embedding': embedding})
    print('Done')


if __name__ == '__main__':
    convert()
