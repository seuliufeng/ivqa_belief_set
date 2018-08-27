import numpy as np


class Reader(object):
    def __init__(self, batch_size=8):
        self._batch_size = batch_size
        self._check_valid_answers()
        self._shuffle_data()

    def _check_valid_answers(self):
        # if True:
        self._valid_ids = np.arange(20).astype(np.int32)
        self._valid_count = self._valid_ids.size
        self._dummy_sample_ids = np.arange(self._batch_size).astype(np.int32)
        self._quest_ids = np.random.choice(100, size=(30,), replace=False)

    def _shuffle_data(self):
        print('Shuffing')
        self._pointer = 0
        np.random.shuffle(self._valid_ids)

    def get_sample_index(self):
        _this_batch_ids = self._pointer + self._dummy_sample_ids
        _pad_mask = _this_batch_ids >= self._valid_count
        _pick_ids = _this_batch_ids % self._valid_count
        index = self._valid_ids[_pick_ids].copy()
        quest_ids = self._quest_ids[index].copy()
        quest_ids[_pad_mask] = -1
        self._pointer += self._batch_size
        if self._pointer >= self._valid_count:
            self._shuffle_data()
        return _pick_ids, quest_ids
        # return _pick_ids, index, quest_ids


def test_reader(batch_size):
    print('Test: batchsize=%d', batch_size)
    reader = Reader(batch_size)
    for i in range(10):
        index, quest_ids = reader.get_sample_index()
        print('Batch %d' % i)
        print('Index:')
        print(index)
        print('Qusetion Indices:')
        print(quest_ids)
    print('\n')


if __name__ == '__main__':
    test_reader(8)
    test_reader(10)

