from util import load_hdf5


class RerankContext(object):
    def __init__(self, subset='trainval', num_cands=5):
        self.subset = subset
        self.is_testing = 'train' not in subset
        self.num_cands = num_cands
        self._load_data()

    def _load_data(self):
        if self.is_testing:
            fpath = '../iccv_vaq/data/sparse_vqa_scores_%s_5nogt.h5' % self.subset
        else:
            fpath = '../iccv_vaq/data/sparse_vqa_scores_%s.h5' % self.subset
        d = load_hdf5(fpath)
        quest_ids = d['quest_ids']
        self.cand_labels = d['top_k_index'][:, :self.num_cands]
        self.vqa_scores = d['top_k_scores'][:, :self.num_cands]
        qid2index = {qid: i for i, qid in enumerate(quest_ids)}
        self.quest_id2index = qid2index
        self.quest_ids = quest_ids

        if self.is_testing:
            fpath = '../iccv_vaq/data/sparse_ivqa_cond_scores%s_5nogt.h5' % self.subset
        else:
            fpath = '../iccv_vaq/data/sparse_ivqa_cond_scores%s.h5' % self.subset
        d = load_hdf5(fpath)
        tmp_ivqa_scores = d['ivqa_scores'][:, :self.num_cands]
        tmp_quest_ids = d['quest_ids']
        tmp_qid2index = {qid: i for i, qid in enumerate(tmp_quest_ids)}
        tmp_order = [tmp_qid2index[qid] for qid in self.quest_ids]
        self.ivqa_scores = tmp_ivqa_scores[tmp_order]

    def get_scores(self, quest_ids):
        index = [self.quest_id2index[qid] for qid in quest_ids]
        b_ivqa_scores = self.ivqa_scores[index]
        b_vqa_scores = self.vqa_scores[index]
        b_cands = self.cand_labels[index]
        return b_cands, b_vqa_scores, b_ivqa_scores



def test_rerank():
    import numpy as np
    rerank_ctx = RerankContext(subset='val')
    quest_ids = rerank_ctx.quest_ids
    b_qids = np.random.choice(quest_ids, size=(10), replace=False)
    cands, vqa_scs, ivqa_scs = rerank_ctx.get_scores(b_qids)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    test_rerank()

