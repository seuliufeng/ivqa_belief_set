import numpy as np
import os
from util import load_hdf5, load_json, get_feature_root, find_image_id_from_fname
from multiprocessing import Process, Queue
from readers.curriculum_sampler import CurriculumSampler

_DATA_ROOT = './'


class AttentionDataReader(object):
    def __init__(self, batch_size=32, subset='kptrain',
                 model_name='', feat_type='res5c',
                 version='v2', counter_sampling=False,
                 n_process=2):
        self._data_queue = None
        self._model_name = model_name
        # self._n_process = n_process
        self._n_process = 1
        self._prefetch_procs = []
        self._batch_size = batch_size
        self._subset = subset
        self._num_top_ans = 2000
        self._prev_index = None
        self._feat_type = feat_type
        self._version_suffix = 'v2_' if version == 'v2' else ''
        self._counter_sampling = counter_sampling

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
                                           counter_sampling=self._counter_sampling)
            proc.start()
            self._prefetch_procs.append(proc)

        def cleanup():
            print 'Terminating BlobFetcher'
            for proc in self._prefetch_procs:
                proc.terminate()
                proc.join()

        import atexit
        atexit.register(cleanup)

    def stop(self):
        print 'Terminating BlobFetcher'
        for proc in self._prefetch_procs:
            proc.terminate()
            proc.join()

    def pop_batch(self):
        return self._data_queue.get()


class AttentionDataPrefetcher(Process):
    def __init__(self, output_queue, proc_id,
                 batch_size=32, subset='trainval',
                 feat_type='res5c', version_suffix='',
                 counter_sampling=False):
        super(AttentionDataPrefetcher, self).__init__()
        self._batch_size = batch_size
        self._proc_id = proc_id
        self._num_top_ans = 2000
        self._queue = output_queue
        self._version_suffix = version_suffix
        self._subset = subset
        self._images = None
        self._quest_len = None
        self._counter_sampling = counter_sampling
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
        data_file = 'data/vqa_std_mscoco_multilabel_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._image_ids = d['image_ids']
        self._feat = d['features']
        self._labels = d['labels']
        self._num = self._labels.shape[0]

    def pop_batch(self):
        index = np.random.choice(self._num, size=(self._batch_size,),
                                 replace=False)
        feats = self._slice_image(index)
        labels = self._slice_answers(index)
        outputs =[feats, labels]
        return outputs

    def get_next_batch(self):
        batch_data = self.pop_batch()
        self._queue.put(batch_data)

    def run(self):
        print('DataFetcher started')
        np.random.seed(self._proc_id)
        # very important, it ensures multiple processes didn't generate the same index
        while True:
            self.get_next_batch()

    def _slice_answers(self, index):
        return self._labels[index]

    def _slice_image(self, index):
        # a = self._slice_attributes(index)
        f = self._slice_global_feature(index)
        return f
        # return np.concatenate([a, f], axis=1)

    def _slice_global_feature(self, index):
        return self._feat[index]


class AttentionFetcher(object):
    def __init__(self, batch_size, subset='kptest',
                 feat_type='res5c', version='v2'):
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
        data_file = 'data/vqa_std_mscoco_multilabel_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._image_ids = d['image_ids']
        self._feat = d['features']
        self._labels = d['labels']
        self._num = self._labels.shape[0]

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num - self._idx)
        index = self._index[self._idx:self._idx + this_batch_size]
        self._idx += this_batch_size
        return index

    def _get_rand_index(self):
        return np.random.choice(self._valid_ids,
                                size=self._batch_size,
                                replace=False)

    def get_index(self):
        if self._is_test:
            return self._get_sequencial_index()
        else:
            return self._get_rand_index()

    def get_test_batch(self):
        index = self.get_index()
        image_id = self._image_ids[index]
        feats = self._slice_image(index)
        a = self._slice_answers(index)
        return feats, a, image_id, image_id

    def _slice_answers(self, index):
        return self._labels[index]

    def _slice_image(self, index):
        # a = self._slice_attributes(index)
        return self._slice_global_feature(index)
        # return np.concatenate([a, f], axis=1

    def _slice_global_feature(self, index):
        return self._feat[index]


if __name__ == '__main__':
    from time import time
    from inference_utils.question_generator_util import SentenceGenerator

    to_sentence = SentenceGenerator(trainset='trainval')
    reader = AttentionDataReader(batch_size=4, subset='trainval', counter_sampling=True)
    reader.start()
    from time import sleep

    t = time()
    for i in range(4):
        data = reader.pop_batch()
        data[0].mean()
        feats, q, q_len, a = data
        for c in range(2):
            q1 = to_sentence.index_to_question(q[c])
            a1 = to_sentence.index_to_top_answer(a[c])
            q2 = to_sentence.index_to_question(q[c + 2])
            a2 = to_sentence.index_to_top_answer(a[c + 2])
            if a1 == 2000 or a2 == 2000:
                continue
            print('Index: %d' % i)
            print('Q1: %s\nA1: %s \n' % (q1, a1))
            print('Q2: %s\nA2: %s \n' % (q2, a2))
            print('\n')
            sleep(0.4)
        # print(data[1].mean())
        # print(data[2].max())
        # print(data[0].shape)

        reader.update_loss(np.random.rand(4))

    avg_time = (time() - t) / 100.
    print('run 100 batches, avg time: %0.2f sec/batch' % avg_time)
    reader.stop()
