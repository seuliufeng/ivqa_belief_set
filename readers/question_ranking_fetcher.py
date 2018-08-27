import numpy as np
import os
from util import load_hdf5, load_json, get_feature_root, \
    find_image_id_from_fname, load_feature_file_vqabaseline
from multiprocessing import Process, Queue
# from readers.curriculum_sampler import CurriculumSampler

_DATA_ROOT = './'
data_root = 'data/'


class Reader(object):
    def __init__(self, batch_size=32, subset='kptrain',
                 model_name='', feat_type='res5c',
                 version='v1', delta=0.2):
        self._data_queue = None
        self._index_queue = None
        self._model_name = model_name
        self._n_process = 1
        self._prefetch_procs = []
        self._batch_size = batch_size
        self._subset = subset
        self._num_top_ans = 2000
        self._prev_index = None
        self._feat_type = feat_type
        self._version_suffix = 'v2_' if version == 'v2' else ''
        self._delta = delta

    def start(self):
        self._data_queue = Queue(10)
        # make index queue larger to ensure loader fully operational
        for proc_id in range(self._n_process):
            proc = AttentionDataPrefetcher(self._data_queue,
                                           proc_id,
                                           self._batch_size,
                                           self._subset,
                                           feat_type=self._feat_type,
                                           version_suffix=self._version_suffix,
                                           delta=self._delta)
            proc.start()
            self._prefetch_procs.append(proc)

        def cleanup():
            print('Terminating BlobFetcher')
            for proc in self._prefetch_procs:
                proc.terminate()
                proc.join()

        import atexit
        atexit.register(cleanup)

    def stop(self):
        print('Terminating BlobFetcher')
        for proc in self._prefetch_procs:
            proc.terminate()
            proc.join()

    def pop_batch(self):
        data = self._data_queue.get()
        return data


class AttentionDataPrefetcher(Process):
    def __init__(self, output_queue, proc_id,
                 batch_size=32, subset='trainval',
                 feat_type='res5c', version_suffix='',
                 delta=0.2):
        super(AttentionDataPrefetcher, self).__init__()
        self._batch_size = batch_size
        self._proc_id = proc_id
        self._num_top_ans = 2000
        self._queue = output_queue
        self._version_suffix = version_suffix
        self._subset = subset
        self._delta = delta
        self._images = None
        self._quest_len = None
        self._quest = None
        self._answer = None
        self.quest_index2counter_index = None
        self._num = None
        self._valid_ids = None
        self._load_data()
        self._feat_type = feat_type.lower()
        self._FEAT_ROOT = get_feature_root(self._subset, self._feat_type)
        self._transpose_feat = 'res5c' in self._feat_type

    def _load_data(self):
        # load meta
        data_file = 'result/cst_ranking_kptrain_delta%g.data' % self._delta
        d = load_hdf5(data_file)
        self._quest_w2v = d['question_w2v']
        self._answer_w2v = d['answer_w2v']
        self._image_ids = d['image_ids']
        self._quest_ids = d['quest_ids']
        self._cst_pairs = d['cst_pairs']

        self._num = self._quest_ids.size
        self._load_global_image_feature()

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        self._feat = d['features']
        self.image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}

    def pop_batch(self):
        index = self._get_question_index()
        # collect pairs
        data_index = self._cst_pairs[index]
        pos_ind = data_index[:, 0].flatten()
        neg_ind = data_index[:, 1].flatten()
        quest_ind = np.concatenate([pos_ind, neg_ind], axis=0)
        image_ids = self._image_ids[quest_ind]
        image_ind = np.array([self.image_id2att_index[t] for t in image_ids],
                             dtype=np.int32)
        # slice
        feats = self._feat[image_ind]
        quest_w2v = self._quest_w2v[quest_ind]
        answer_w2v = self._answer_w2v[quest_ind]
        outputs = [feats, quest_w2v, answer_w2v]
        return outputs

    def _get_question_index(self):
        index = np.random.choice(self._num, size=(self._batch_size,),
                                 replace=False)
        return index

    def get_next_batch(self):
        batch_data = self.pop_batch()
        self._queue.put(batch_data)

    def run(self):
        print('DataFetcher started')
        np.random.seed(self._proc_id)
        # very important, it ensures multiple processes didn't generate the same index
        while True:
            self.get_next_batch()


class TestReader(object):
    def __init__(self, batch_size, subset='kptest',
                 feat_type='res5c', version='v1',
                 use_fb_data=False):
        self._batch_size = batch_size
        self._num_top_ans = 2000
        self._subset = subset
        self._images = None
        self._quest_len = None
        self._quest_ids = None
        self._quest = None
        self._mask = None
        self._answer = None
        self._num = None
        self._valid_ids = None
        self._vqa_image_ids = None
        self.use_fb_data = use_fb_data
        self._version_suffix = 'v2_' if version == 'v2' else ''
        self._load_data()
        self._feat_type = feat_type.lower()
        self._FEAT_ROOT = get_feature_root(self._subset, self._feat_type)
        self._transpose_feat = self._feat_type == 'res5c'
        #
        self._idx = 0
        self._index = np.arange(self._num)
        self._is_test = True

    @property
    def num_batches(self):
        from math import ceil
        n = ceil(self._num / float(self._batch_size))
        return int(n)

    def start(self):
        pass

    def stop(self):
        pass

    def _load_data(self):
        # load meta
        data_file = 'result/cst_ranking_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._quest_w2v = d['question_w2v']
        self._answer_w2v = d['answer_w2v']
        self._image_ids = d['image_ids']
        self._quest_ids = d['aug_quest_ids']
        self._cider_scs = d['scores']

        self._num = self._quest_ids.size
        self._load_global_image_feature()

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        self._feat = d['features']
        self.image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num - self._idx)
        index = self._index[self._idx:self._idx + this_batch_size]
        self._idx += this_batch_size
        return index

    def get_test_batch(self):
        quest_ind = self._get_sequencial_index()
        quest_ids = self._quest_ids[quest_ind]
        cider_sc = self._cider_scs[quest_ind]
        # collect pairs
        image_ids = self._image_ids[quest_ind]
        image_ind = np.array([self.image_id2att_index[t] for t in image_ids],
                             dtype=np.int32)
        # slice
        feats = self._feat[image_ind]
        quest_w2v = self._quest_w2v[quest_ind]
        answer_w2v = self._answer_w2v[quest_ind]
        return feats, quest_w2v, answer_w2v, cider_sc, quest_ids, image_ids


if __name__ == '__main__':
    from time import time
    from inference_utils.question_generator_util import SentenceGenerator

    to_sentence = SentenceGenerator(trainset='trainval')
    reader = AttentionDataPrefetcher(batch_size=4, subset='kptrain')
    for i in range(10):
        print(i)
        data = reader.pop_batch()
