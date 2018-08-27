from util import load_json, save_json

question_items = ["what", "where", "who", "why", "does"]
# nn_items = ["type", "kind", "color", "picture", "photo", "image"]
# adjective_items = ["visible"]

_black_list = ['do', 'have', 'be', 'it'] + question_items
_black_list = {k: None for k in _black_list}


def load_vocab():
    with open('data/vqa_trainval_question_word_counts.txt', 'r') as fs:
        lines = fs.readlines()
        words = [line.split()[0].strip() for line in lines]
    return {word: i for i, word in enumerate(words)}


class Vocabulary(object):
    def __init__(self):
        self._word2idx = load_vocab()
        self.unk_id = len(self._word2idx)

    def word2id(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self.unk_id


def process_visual_facts():
    vocab = Vocabulary()
    d = load_json('/usr/data/fl302/code/premise17/premises/coco_spice_outputs_full.json')
    vis_facts = {}
    for i, item in enumerate(d):
        if i % 10000 == 0:
            print('processed %d/%d' % (i, len(d)))
        image_id = item['image_id']
        elems = []
        for _tmp in item['ref_tuples']:
            elems += _tmp['tuple']
        _token_ids = [vocab.word2id(e) for e in elems if e not in _black_list]
        if len(_token_ids) == 0:
            print('Warning: empty for image_id: %s' % image_id)
            print(elems)
        vis_facts[image_id] = _token_ids
    save_json('data/visual_facts_tokened_val.json', {'vis_facts': vis_facts})


if __name__ == '__main__':
    process_visual_facts()
