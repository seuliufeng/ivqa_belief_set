import numpy as np
import os
from util import load_hdf5, load_json, get_feature_root, find_image_id_from_fname
from multiprocessing import Process, Queue
from readers.curriculum_sampler import CurriculumSampler

_DATA_ROOT = './'
_DATA_ROOT = './'


class AttentionDataReader(object):
    def __init__(self, batch_size=32, subset='kptrain',
                 model_name='', epsilon=0.5, feat_type='res5c',
                 version='v2', counter_sampling=False):
        self._data_queue = None
        self._index_queue = None
        self._model_name = model_name
        self._epsilon = epsilon
        self._n_process = 4
        self._prefetch_procs = []
        self._batch_size = batch_size
        self._subset = subset
        self._num_top_ans = 2000
        self._prev_index = None
        self._feat_type = feat_type
        self._version_suffix = 'v2_' if version == 'v2' else ''
        self._counter_sampling = counter_sampling
        self._create_sampler()

    def _create_sampler(self):
        data_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.data' %
                                 (self._version_suffix, self._subset))
        d = load_hdf5(data_file)
        answer = d['answer'].astype(np.int32)
        sampler_batch_size = int(self._batch_size / 2) if self._counter_sampling \
            else self._batch_size
        self._sampler = CurriculumSampler(batch_size=sampler_batch_size,
                                          num_samples=answer.size,
                                          epsilon=self._epsilon,
                                          suffix=self._model_name)
        valid_ids = np.where(answer < self._num_top_ans)[0]
        self._sampler.set_valid_index(valid_ids)

    def fill_index_queue(self):
        # t = time()
        # idx = 0
        while not self._index_queue.full():
            index = self._sampler.sample_batch()
            self._index_queue.put(index)
            #     idx += 1
            # print('\n Added %d works in %0.2f sec.' % (idx, time()-t))

    def backup_statistics(self):
        self._sampler.backup_statistics()

    def start(self):
        self._data_queue = Queue(10)
        self._index_queue = Queue(20)
        # make index queue larger to ensure loader fully operational
        for proc_id in range(self._n_process):
            proc = AttentionDataPrefetcher(self._data_queue,
                                           self._index_queue,
                                           proc_id,
                                           self._batch_size,
                                           self._subset,
                                           feat_type=self._feat_type,
                                           version_suffix=self._version_suffix,
                                           counter_sampling=self._counter_sampling)
            proc.start()
            self._prefetch_procs.append(proc)
            self.fill_index_queue()

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

    def update_loss(self, loss=None):
        self._sampler.update_loss(self._prev_index, loss)

    def pop_batch(self):
        self.fill_index_queue()
        data = self._data_queue.get()
        self._prev_index = data[0]
        return data[1:]


class AttentionDataPrefetcher(Process):
    def __init__(self, output_queue, index_queue, proc_id,
                 batch_size=32, subset='trainval',
                 feat_type='res5c', version_suffix='',
                 counter_sampling=False):
        super(AttentionDataPrefetcher, self).__init__()
        self._batch_size = batch_size
        self._proc_id = proc_id
        self._num_top_ans = 2000
        self._queue = output_queue
        self._version_suffix = version_suffix
        self._index_queue = index_queue
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
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.meta' %
                                 (self._version_suffix, self._subset))
        data_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.data' %
                                 (self._version_suffix, self._subset))
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        self._quest_ids = np.array(d['quest_id'])
        # load QA data
        d = load_hdf5(data_file)
        self._load_counter_examples(d)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['answer'].astype(np.int32)
        self._check_valid_answers()

    def _load_counter_examples(self, d):
        if self._counter_sampling:
            counter_examples = d['counter_example']
            quest_id2index = {qid: id for id, qid in enumerate(d['quest_ids'])}
            counter_index = np.array([quest_id2index[c_eg] if c_eg in quest_id2index
                                      else -1 for c_eg in counter_examples])
            self.quest_index2counter_index = counter_index
            print('Data Fetcher: find %d/%d counter examples' %
                  (np.sum(counter_index > 0), counter_index.size))

    def _check_valid_answers(self):
        self._valid_ids = np.where(self._answer < self._num_top_ans)[0]
        print('%d/%d' % (self._valid_ids.size, self._answer.size))

    def pop_batch(self):
        index = self._get_question_index()
        feats = self._load_image_features(index)
        q, q_len = self._slice_questions(index)
        a = self._slice_answers(index)
        outputs = [index, feats, q, q_len, a]
        return outputs

    def _get_question_index(self):
        index = self._index_queue.get()
        if self._counter_sampling:
            counter_index = self.quest_index2counter_index[index]
            counter_index = counter_index[counter_index > 0]
            # add random samples to compensate samples without counter example
            num_miss = index.size - counter_index.size
            missed = np.random.choice(self._valid_ids,
                                      size=num_miss,
                                      replace=False)  #
            index = np.concatenate([index, counter_index, missed])
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

    def _load_image_features(self, index):
        feats = []
        for idx in index:
            filename = self._images[idx]
            try:
                f = np.load(os.path.join(self._FEAT_ROOT, filename + '.npz'))['x']
                if self._transpose_feat:
                    feats.append(f.transpose((1, 2, 0))[np.newaxis, ::])
                else:
                    feats.append(f[np.newaxis, ::])
            except Exception, e:
                print('Process %d: Error loading file: %s' % (self._proc_id,
                                                              os.path.join(self._FEAT_ROOT,
                                                                           filename + '.npz')))
                print(str(e))
                feats.append(np.zeros([1, 14, 14, 2048], dtype=np.float32))

        return np.concatenate(feats, axis=0).astype(np.float32)

    def _slice_questions(self, index):
        q_len = self._quest_len[index]
        max_len = q_len.max()
        seq = self._quest[index, :max_len]
        return seq, q_len

    def _slice_answers(self, index):
        return self._answer[index]


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
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.meta' %
                                 (self._version_suffix, self._subset))
        data_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.data' %
                                 (self._version_suffix, self._subset))
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)
        self._quest_ids = np.array(d['quest_id'])
        # load QA data
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['answer'].astype(np.int32)
        self._check_valid_answers()
        self._num = self._quest_len.size

    def _check_valid_answers(self):
        self._valid_ids = np.where(self._answer < self._num_top_ans)[0]

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
        image_id = self._vqa_image_ids[index]
        feats = self._load_image_features(index)
        q, q_len = self._slice_questions(index)
        a = self._slice_answers(index)
        quest_id = self._quest_ids[index]
        return feats, q, q_len, a, quest_id, image_id

    def pop_batch(self):
        index = self.get_index()
        feats = self._load_image_features(index)
        q, q_len = self._slice_questions(index)
        a = self._slice_answers(index)
        return feats, q, q_len, a

    def _load_image_features(self, index):
        feats = []
        for idx in index:
            filename = self._images[idx]
            f = np.load(os.path.join(self._FEAT_ROOT, filename + '.npz'))['x']
            if self._transpose_feat:
                feats.append(f.transpose((1, 2, 0))[np.newaxis, ::])
            else:
                feats.append(f[np.newaxis, ::])
        return np.concatenate(feats, axis=0).astype(np.float32)

    def _slice_questions(self, index):
        q_len = self._quest_len[index]
        max_len = q_len.max()
        seq = self._quest[index, :max_len]
        return seq, q_len

    def _slice_answers(self, index):
        return self._answer[index]


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
            q2 = to_sentence.index_to_question(q[c+2])
            a2 = to_sentence.index_to_top_answer(a[c+2])
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
