import numpy as np


class IrrelevantManager(object):
    def __init__(self, image_ids):
        self.name = 'IRGenerator'
        self.image_ids = np.array(image_ids, dtype=np.int32)
        # load image neighbours
        self._load_nn_info()
        self._build_image_id2index()
        # load blacklist
        self._load_black_list()

    def _build_image_id2index(self):
        image_id2index = {}
        for i, image_id in enumerate(self.image_ids):
            if image_id in image_id2index:
                image_id2index[image_id].append(i)
            else:
                image_id2index[image_id] = [i]
        self.image_id2index = image_id2index

    def _load_nn_info(self):
        print('%s: Loading nearest neighbours' % self.name)
        from scipy.io import loadmat
        d = loadmat('/data1/fl302/projects/compute_nn/image_ids.mat')
        image_ids = d['image_ids'].flatten().astype(np.int32)
        d = loadmat('/data1/fl302/projects/compute_nn/TrainvalKnnInfo.mat')
        image_nn = d['Inds']
        self.image_id2nn = {}
        for image_id, nns in zip(image_ids, image_nn):
            nn_im_ids = image_ids[nns - 1]  # convert relative to absolute
            self.image_id2nn[image_id] = {im_id: None for im_id in nn_im_ids}
        unk_image_ids_loaded = np.unique(self.image_ids)
        self.unk_image_ids = unk_image_ids_loaded
        blacklist_keys = np.setdiff1d(image_ids, unk_image_ids_loaded)
        self.black_list = {int(im_id): None for im_id in blacklist_keys}

    def _load_black_list(self):
        from util import load_json
        test_black_list = load_json('data/kptest_blacklist.json')
        self.black_list.update(test_black_list)
        print('%s: %d keys in blacklist' % (self.name, len(self.black_list)))

    def query_irrelevant(self, quest_index):
        image_ids = self.image_ids[quest_index]
        # query irrelevant corresponds to these
        neg_quest_inds = self._query_by_image_ids(image_ids)
        return np.array(neg_quest_inds, dtype=np.int32)

    def _query_by_image_ids(self, image_ids):
        neg_quest_inds = []
        for image_id in image_ids:
            assert (image_id not in self.black_list)
            # sample a negative image
            neg_image_id = self._sample_negative_image(image_id)
            # sample a question given image_id
            qindex = self._sample_question_for_image(neg_image_id)
            neg_quest_inds.append(qindex)
        return neg_quest_inds

    def _sample_negative_image(self, image_id):
        nn_ids = self.image_id2nn[image_id]
        while True:
            _idx = np.random.choice(self.unk_image_ids)  # this is positive my love
            if _idx in nn_ids:
                continue
            if _idx not in self.black_list:
                return _idx

    def _sample_question_for_image(self, image_id):
        inds = self.image_id2index[image_id]
        return np.random.choice(inds)


def test_irrelevant_management():
    from util import load_json, find_image_id_from_fname
    d = load_json('data/vqa_std_mscoco_trainval.meta')
    image_ids = [find_image_id_from_fname(fname) for fname in d['images']]
    irrel_man = IrrelevantManager(image_ids)
    num = len(image_ids)

    batch_size = 4
    num_batches = 4
    from time import time
    t = time()
    for i in range(num_batches):
        print('\nBatch %d: ' % i)
        index = np.random.choice(num, size=(batch_size,), replace=False)
        neg_index = irrel_man.query_irrelevant(index)
        print('pos:')
        print(index)
        print('neg:')
        print(neg_index)
    tot = time() - t
    print('Time: %0.2f ms/batch' % (tot * 1000. / float(num_batches)))


if __name__ == '__main__':
    test_irrelevant_management()
