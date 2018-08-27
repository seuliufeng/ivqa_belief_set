from util import load_json
import numpy as np
from post_process_variation_questions import put_to_array
import pdb


def _random_pick(elems):
    idx = np.random.randint(len(elems))
    return elems[idx]


class VQABufferManager(object):
    def __init__(self, filename='vqa_replay_buffer/vqa_replay.json'):
        d = load_json(filename)
        self._parse_buffer(d['memory'])

    def _parse_buffer(self, memory):
        self.memory = {}
        self.empty_keys = {}
        self.non_empty_keys = []
        for i, quest_key in enumerate(memory.keys()):
            quest_id = int(quest_key)
            pathes = memory[quest_key]
            if len(pathes) == 0:
                self.memory[quest_id] = []
                self.empty_keys[quest_id] = None
                continue

            _quests = []
            for p in pathes.keys():
                _tokens = [int(t) for t in p.split(' ')]
                _quests.append(_tokens[1:-1])  # remove start and end token
            self.memory[quest_id] = _quests
            self.non_empty_keys.append(quest_id)
        self.non_empty_keys_dict = {k: None for k in self.non_empty_keys}

    def query(self, quest_ids):
        pathes = []
        mask = []
        for quest_id in quest_ids:
            is_valid, p = self._query_one(quest_id)
            pathes.append(p)
            mask.append(is_valid)
        arr, arr_len = put_to_array(pathes, pad_token=0)
        mask = np.array(mask, dtype=np.float32)
        return mask, arr, arr_len

    def _query_one(self, quest_id):
        try:
            # is_valid = False if quest_id in self.empty_keys else True
            # query_key = _random_pick(self.non_empty_keys) \
            #     if quest_id in self.empty_keys else quest_id
            is_valid = quest_id in self.non_empty_keys_dict
            query_key = quest_id if is_valid else _random_pick(self.non_empty_keys)
            return is_valid, _random_pick(self.memory[query_key])
        except:
            print('Error')
            pdb.set_trace()


def test_random_pick():
    print('\nTest 1')
    arr = ['a', 'b', 'c', 'd', 'e']
    d = {k: 0 for k in arr}
    for i in range(10000):
        s = _random_pick(arr)
        d[s] += 1
    for k, v in d.iteritems():
        print('%s: %d' % (k, v))

    print('\nTest 2')
    arr = [['a', 'b'], ['c', 'd'], ['e']]
    for i in range(10):
        print(_random_pick(arr))


def test_vqa_context():
    mem = VQABufferManager()
    non_emp = mem.non_empty_keys
    emp = mem.empty_keys.keys()

    _non_emp = [_random_pick(non_emp) for _ in range(5)]
    _emp = [_random_pick(emp) for _ in range(5)]
    mask, arr, arr_len = mem.query(_non_emp + _emp)
    import pdb
    pdb.set_trace()
    print(mask)
    print(arr)
    print(arr_len)


if __name__ == '__main__':
    # test_random_pick()
    test_vqa_context()
