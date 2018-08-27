import numpy as np


def serialize_path(path):
    return ' '.join([str(t) for t in path])


def parse_gt_questions(capt, capt_len):
    seqs = []
    for c, clen in zip(capt, capt_len):
        seqs.append(c[:clen].tolist() + [2])
    return seqs


class UniqueReward(object):
    def __init__(self, init=True):
        self.history = {}
        self.num_total = float(0)
        self.num_unique = float(0)
        self.average_count = 0.0
        self.count_reference = 0.0
        self.average_reference = 0.0
        self.iter = 0.0
        self.t = 0.01
        if init:
            self.initialise_with_gt()

    def initialise_with_gt(self):
        # load data
        from util import load_hdf5
        from time import time
        t = time()
        print('Initialising statastics with ground truth')
        d = load_hdf5('data/vqa_std_mscoco_kprestval.data')
        gts = parse_gt_questions(d['quest_arr'], d['quest_len'])
        # update stat
        self._update_samples(gts, generate_key=True, update_reference=True)

        print('Initialisation finished, time %0.2fs' % (time() - t))
        print('Total number of questions: %d' % self.num_total)
        print('Number of unique questions: %d' % self.num_unique)
        print('Average question counts: %0.2f' % self.average_count)
        num_counts = self.history.values()
        max_count = max(num_counts)
        min_count = min(num_counts)
        for t in [0.01, 0.02, 0.03, 0.05]:
            r_min = np.exp(-max_count / self.average_count * t)
            r_max = np.exp(-min_count / self.average_count * t)
            print('[t=%0.4f] Max question counts: %d, ratio %0.3f, estimated reward % 0.3f' % (
                t, max_count, max_count / self.average_count, r_min))
            print('[t=%0.4f] Min question counts: %d, ratio %0.3f, estimated reward: %0.3f' % (
                t, min_count, min_count / self.average_count, r_max))
        self.gt_keys = {k: None for k in self.history.keys()}

    def _update_samples(self, samples, generate_key=False, update_reference=False):
        for _key in samples:
            if generate_key:
                _key = serialize_path(_key)
            if _key in self.history:
                self.history[_key] += 1.
            else:
                self.history[_key] = 1.
                self.num_unique += 1.
            self.num_total += 1.0
        # update average
        if update_reference:
            self.count_reference = self.num_total
            self.average_count = self.num_total / self.num_unique
            self.average_reference = self.average_count
        else:
            self.iter += len(samples)
            self.average_count = self.average_reference * (1 + self.iter / self.count_reference)

    def get_reward(self, samples):
        rewards = []
        samples_keys = []
        is_gt = []
        for ps in samples:
            for p in ps:
                _key = serialize_path(p)
                _c = self._path_query(_key)
                rewards.append(_c)
                is_gt.append(_key in self.gt_keys)
                samples_keys.append(_key)
        rewards = np.array(rewards, dtype=np.float32) / self.average_count
        rewards = np.exp(-rewards * self.t)
        is_gt = np.array(is_gt, dtype=np.bool)
        # update stats
        self._update_samples(samples_keys, generate_key=False)
        return rewards, is_gt

    def _path_query(self, _key):
        _count = 0.
        if _key in self.history:
            _count = self.history[_key]
        return _count


def test_unique_reward():
    env = UniqueReward()
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    test_unique_reward()
