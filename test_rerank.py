from util import load_hdf5, save_hdf5
from inference_utils import vocabulary
from nltk.tokenize import word_tokenize
import numpy as np


def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


def _load_top_answers(vocab_file):
    print('Loading top answer vocabulary...')
    with open(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
        top_ans_vocab = [line.strip() for line in reverse_vocab]
    return top_ans_vocab


def encode_to_sequence(vocab, seqs):
    t_seqs = []
    # _max_ans_vocab_size = len(vocab)
    for sentence in seqs:
        tokenized = _tokenize_sentence(sentence) if type(sentence) != list else sentence
        ids = [vocab.word_to_id(word) for word in tokenized]
        t_seqs.append(ids)
    return t_seqs


def encode_top_answers(vocab):
    # load top 2000 answer vocabulary
    top_ans_vocab = _load_top_answers('data/vqa_trainval_top2000_answers.txt')
    return encode_to_sequence(vocab, top_ans_vocab)

# encode top 2000 answers
_vocab = vocabulary.Vocabulary('data/vqa_trainval_question_answer_word_counts.txt')
top_answer_seqs = encode_top_answers(_vocab)

# load questions of the validation set
import json
ann_file = 'data/annotations/OpenEnded_mscoco_val2014_questions.json'
d = json.load(open(ann_file, 'r'))['questions']
qid2q = {info['question_id']: info['question'] for info in d}
qid2image_id = {info['question_id']: info['image_id'] for info in d}

dd = load_hdf5('vqa_att_g1_scores.h5')
# encode question to sequence
quest_ids = dd['quest_ids']
quests, qid2index = [], {}
image_ids = []
for i, qid in enumerate(quest_ids):
    quests.append(qid2q[qid])
    qid2index[qid] = i
    image_ids.append(qid2image_id[qid])
quest_seqs = encode_to_sequence(_vocab, quests)


# merge questions to a matrix
def _mount_to_matrix(seqs):
    seq_len = [len(q) for q in seqs]
    max_len = max(seq_len)
    num = len(seq_len)
    quest_arr = np.zeros([num, max_len], dtype=np.int32)
    for i, x in enumerate(quest_arr):
        x[:seq_len[i]] = seqs[i]
    quest_len = np.array(seq_len, dtype=np.int32)
    return quest_arr, quest_len


quest_arr, quest_len = _mount_to_matrix(quest_seqs)
cand_arr, cand_len = _mount_to_matrix(top_answer_seqs)
# get index of predicted answers
scores = dd['scores']
cand_answer_index = (-scores).argsort(axis=1)[:, :10]

print('Saving...')
# save results
save_hdf5('data/att_rerank/qa_arr.h5', {'quest_arr': quest_arr,
                                        'quest_len': quest_len,
                                        'att_cand_arr': cand_answer_index,
                                        'top2000_arr': cand_arr,
                                        'top2000_len': cand_len,
                                        'quest_id': quest_ids,
                                        'image_id': np.array(image_ids, dtype=np.int32)})
print('All done!')

