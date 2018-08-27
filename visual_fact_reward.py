import numpy as np
from scipy.io import loadmat
from util import load_json
import pdb

_EPS = 1e-8


class VisualFactReward(object):
    def __init__(self):
        word2lemma = loadmat('data/quest_token2lemma.mat')['word2lemma']
        self.word2lemma = word2lemma.flatten()
        self.valid_vocab_size = self.word2lemma.size
        self._load_visual_facts()
        self._load_mapping()

    def _load_visual_facts(self):
        # d = load_json('data/visual_facts_tokened.json')['vis_facts']
        d = load_json('data/visual_facts_tokened_val.json')['vis_facts']
        gts = {}
        for image_id in d:
            v = set(d[image_id])
            gts[image_id] = v
        self.gt_facts = gts

    def _load_mapping(self):
        d = load_json('vqa_val_quest_id2im_id.json')
        mapping = {}
        for k, v in d.iteritems():
            mapping[int(k)] = u'%d' % v
        self.quest_id2image_id = mapping
        self.dummy_quest_id = self.quest_id2image_id.keys()[0]

    def get_image_ids(self, quest_ids):
        return [self.quest_id2image_id[qid] for qid in quest_ids]

    def get_reward(self, sampled, quest_ids):
        pad_tab = quest_ids == -1
        quest_ids[pad_tab] = self.dummy_quest_id
        image_ids = self.get_image_ids(quest_ids)
        # normalise inputs
        scores = []
        for ps, image_id in zip(sampled, image_ids):
            scores += self.compute_score(ps, image_id)
        scores = np.array(scores, np.float32)
        scores[pad_tab] = 0
        return scores

    def compute_score(self, ps, image_id):
        gt = self.gt_facts[image_id]  # it is a set
        gt_size = float(len(gt) + _EPS)
        scores = []
        for path in ps:
            lem_path = set(self.word2lemma[t] for t in path if t < self.valid_vocab_size)
            intersect = (lem_path & gt)
            recall = len(intersect) / gt_size
            scores.append(recall)
        return scores


def test_fact_reward():
    import nltk
    from lemmatise_visual_facts import Vocabulary
    vocab = Vocabulary()
    vf = VisualFactReward()
    image_id = 194200

    def _process_input(q):
        tokens = nltk.tokenize.word_tokenize(q.lower())
        ids = [vocab.word2id(t) for t in tokens]
        return ids

    rand1 = 'What sport are they playing?'
    rand2 = 'How many bats are in the photo?'
    rand3 = 'How many people are in the photo?'
    pathes = [[_process_input(rand1), _process_input(rand2), _process_input(rand3)]]

    scores = vf.get_reward(pathes, [image_id])
    print(scores)
    pdb.set_trace()


if __name__ == '__main__':
    test_fact_reward()
