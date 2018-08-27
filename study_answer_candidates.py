import os
import numpy as np
from util import load_hdf5
from w2v_answer_encoder import MultiChoiceQuestionManger
import pdb


def study_hit_rate(subset='dev'):
    data_root = '/import/vision-ephemeral/fl302/code/VQA-tensorflow/data'
    data_file = os.path.join(data_root, 'vqa-advrl_vqa_score2000_%s.hdf5' % subset)
    d = load_hdf5(data_file)
    scores = d['confidence']
    labels = d['labels']
    valid = labels != 2000

    scores = scores[valid]
    labels = labels[valid][:, np.newaxis]
    tmp = -scores
    tmp[:, -1] = 0.
    inds = (tmp).argsort(axis=1)

    for i in range(5):
        k = i + 1
        loc_inds = inds[:, :k]
        num_hits = np.equal(loc_inds, labels).sum(axis=1).sum()
        print('Top %d hit rate: %0.2f' % (k, float(num_hits)/scores.shape[0]*100.))



def load_vqa_predictions(subset):
    data_file = '../iccv_vaq/data/sparse_vqa_scores_%s_0.h5' % subset
    data_file = '../iccv_vaq/data/sparse_vqa_scores_dev_5nogt.h5'
    d = load_hdf5(data_file)
    quest_ids = d['quest_ids']
    pdb.set_trace()


    mc_ctx = MultiChoiceQuestionManger(subset='val', load_ans=True)
    type_labels = []
    for qid in quest_ids:
        coding = mc_ctx.get_answer_type_coding(qid)
        type_labels.append(coding)
    type_labels = np.array(type_labels)
    unique_ids = np.unique(type_labels)
    print(np.bincount(type_labels))
    valid_types = [u'other', u'number', u'yes/no']
    yes_no_id = mc_ctx._answer_type2id['yes/no']
    # yes_no_id = mc_ctx._answer_type2id['other']
    # yes_no_id = mc_ctx._answer_type2id['number']

    is_yes_no = type_labels == yes_no_id
    candidates = d['top_k_index'][:, :3]

    num_yes_no = is_yes_no.sum()
    num_tot = candidates.shape[0]
    print('Yes/No: %d in %d' % (num_yes_no, num_tot))
    yes_top_ans_id = 0
    no_top_ans_id = 1

    yes_no_cands = candidates[is_yes_no]
    has_yes = (yes_no_cands == yes_top_ans_id).sum(axis=1) == 1
    print(has_yes.sum())
    has_no = (yes_no_cands == no_top_ans_id).sum(axis=1) == 1
    print(has_no.sum())
    yn_inter = np.logical_and(has_yes, has_no)
    yn_union = np.logical_or(has_yes, has_no)
    print('Either yes/no %d:' % yn_union.sum())
    print('Both yes/no %d:' % yn_inter.sum())


if __name__ == '__main__':
    load_vqa_predictions('dev')
    # study_hit_rate('dev')
    # study_hit_rate('trainval')
