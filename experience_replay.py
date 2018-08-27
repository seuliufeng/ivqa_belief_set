from config import ModelConfig, VOCAB_CONFIG
from post_process_variation_questions import put_to_array
import numpy as np

END_TOKEN = VOCAB_CONFIG.end_token_id
START_TOKEN = VOCAB_CONFIG.start_token_id


class ReplayBuffer(object):
    def __init__(self, batch_size, ratio=2):
        self.ratio = ratio
        self.pad_token = 15954 - 1
        self.thresh = 1.0 / 3.0
        self.batch_size = batch_size
        self.num_pos_in_batch = self.batch_size
        self.num_neg_in_batch = self.batch_size * ratio
        self.min_count = 10000
        self.max_count = 1000000
        self.mix_ratio = 0.5
        self.neg_in_init = int(self.mix_ratio * self.num_neg_in_batch)
        self.neg_in_policy = self.num_neg_in_batch - self.neg_in_init
        self.pos_mem = {}
        self.neg_mem = {}
        # init positive batch
        self._init_exemplars('kptrain')
        self._init_exemplars('kprestval')
        self.num_pos = len(self.pos_mem)
        self.pos_data = [k for k in self.pos_mem.keys()]
        # init negative batch
        from util import load_json
        self.init_neg_data = load_json('data/lm_init_neg_pathes.json')
        self.policy_neg_data = []
        # init labels
        labels = np.zeros(shape=(self.num_pos_in_batch + self.num_neg_in_batch,),
                          dtype=np.float32)
        labels[:self.num_pos_in_batch] = 1.0
        self.labels = labels

    def _init_exemplars(self, subset):
        from util import load_hdf5
        print('Initialising statastics with ground truth')
        d = load_hdf5('data/vqa_std_mscoco_%s.data' % subset)
        gts = self.parse_gt_questions(d['quest_arr'], d['quest_len'])
        # update stat
        self._update_samples(gts, generate_key=True)

    def _update_samples(self, samples, generate_key=False):
        for _key in samples:
            if generate_key:
                _key = self.serialize_path(_key)
            if _key not in self.pos_mem:
                self.pos_mem[_key] = None

    def insert(self, samples, scores):
        for p, sc in zip(samples, scores):
            if sc < self.thresh:
                ser_key = self.serialize_path(p)
                self.policy_neg_data.append(ser_key)
        # remove the first 10% data if cache is full
        if len(self.policy_neg_data) > self.max_count:
            self.policy_neg_data = self.policy_neg_data[int(0.1 * self.max_count):]

    def get_batch(self):
        # random sample #bs positive
        real_pathes = self.random_pick_from_set(self.pos_data, self.num_pos_in_batch)
        real_arr, real_arr_len = put_to_array(real_pathes, pad_token=self.pad_token,
                                              max_length=20)
        # random sample from other negatives
        fake_pathes = []
        num_in_policy = min(len(self.policy_neg_data), self.neg_in_policy)
        # print('Samping %d from Neg Policy' % num_in_policy)
        fake_pathes += self.random_pick_from_set(self.policy_neg_data, num_in_policy)
        # random sample from init negative
        num_in_init = max(self.neg_in_init, self.num_neg_in_batch-num_in_policy)
        # print('Samping %d from Neg Init' % num_in_init)
        fake_pathes += self.random_pick_from_set(self.init_neg_data, num_in_init)
        fake_arr, fake_arr_len = put_to_array(fake_pathes, pad_token=self.pad_token,
                                              max_length=20)
        return [fake_arr, fake_arr_len, real_arr, real_arr_len]

    @staticmethod
    def random_pick_from_set(src, num):
        if num == 0:
            return []
        inds = np.random.choice(len(src), size=(num,),
                                replace=False)
        pathes = []
        for id in inds:
            p = [int(t) for t in src[id].split(' ')]
            pathes.append(p)
        return pathes

    @staticmethod
    def parse_gt_questions(capt, capt_len):
        seqs = []
        for c, clen in zip(capt, capt_len):
            seqs.append(c[:clen].tolist() + [END_TOKEN])
        return seqs

    @staticmethod
    def serialize_path(path):
        return ' '.join([str(t) for t in path])


def test_replay_buffer():
    rbf = ReplayBuffer(batch_size=4, ratio=2)
    arr, arr_len, labels = rbf.get_batch()
    print('Arr:')
    print(arr)
    print('Arr-Len')
    print(arr_len)
    print('Labels')
    print(labels)
    rand_arr = np.random.randint(low=0, high=100, size=(4, 8))
    rand_pathes = [a.tolist() for a in rand_arr]
    scores = np.random.rand(4) * 0.2
    rbf.insert(rand_pathes, scores)
    arr, arr_len, labels = rbf.get_batch()
    print('-------- Insert --------')
    print('Arr:')
    print(arr)
    print('Arr-Len')
    print(arr_len)
    print('Labels')
    print(labels)
    rand_arr = np.random.randint(low=0, high=100, size=(4, 8))
    rand_pathes = [a.tolist() for a in rand_arr]
    scores = np.random.rand(4) * 0.2
    rbf.insert(rand_pathes, scores)
    arr, arr_len, labels = rbf.get_batch()
    print('-------- Insert --------')
    print('Arr:')
    print(arr)
    print('Arr-Len')
    print(arr_len)
    print('Labels')
    print(labels)


if __name__ == '__main__':
    test_replay_buffer()

