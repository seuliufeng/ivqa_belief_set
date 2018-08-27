import numpy as np
import os
from util import load_hdf5, load_json, get_feature_root, find_image_id_from_fname
from multiprocessing import Process, Queue
from nn_management import IrrelevantManager

_DATA_ROOT = './'


def _concat_arrays(arr1, arr2):
    n_arr1, max_d_arr1 = arr1.shape
    n_arr2, max_d_arr2 = arr2.shape
    if max_d_arr1 != max_d_arr2:
        max_d = max(max_d_arr1, max_d_arr2)
        pad_d1 = max_d - max_d_arr1
        pad_d2 = max_d - max_d_arr2
        # padding
        pad_1 = np.zeros([n_arr1, pad_d1], dtype=arr1.dtype)
        arr1 = np.concatenate([arr1, pad_1], 1)
        pad_2 = np.zeros([n_arr2, pad_d2], dtype=arr2.dtype)
        arr2 = np.concatenate([arr2, pad_2], 1)
    # concatenate
    return np.concatenate([arr1, arr2], 0)


class AttentionDataReader(object):
    def __init__(self, batch_size=32, subset='kptrain',
                 model_name='', epsilon=0.5, feat_type='res5c',
                 version='v2', counter_sampling=False):
        self._data_queue = None
        self._index_queue = None
        self._model_name = model_name
        self._epsilon = epsilon
        self._n_process = 2
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
        return self._data_queue.get()


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
        self._rel_batch_size = int(self._batch_size * 0.5)
        self._ez_ir_batch_size = int(self._batch_size * 0.3)
        self._ir_batch_size = self._batch_size - self._rel_batch_size - self._ez_ir_batch_size

    def _load_data(self):
        self._load_irrelevant_data()
        self._load_relevant_data()

    def _load_irrelevant_data(self):
        # load meta
        d = load_json('data/QRD_irrelevant_meta.json')
        self._ir_images = d['images']
        # load QA data
        d = load_hdf5('data/QRD_irrelevant_data.data')
        self._ir_arr = d['quest'].astype(np.int32)
        self._ir_arr_len = d['quest_len'].astype(np.int32)
        self._num_ir = len(self._ir_images)

    def _load_relevant_data(self):
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/vqa_std_mscoco_%s.meta' % self._subset)
        data_file = os.path.join(_DATA_ROOT,
                                 'data/vqa_std_mscoco_%s.data' % self._subset)
        # load meta
        d = load_json(meta_file)
        self._r_images = d['images']
        image_ids = [find_image_id_from_fname(fname) for fname in self._r_images]
        self.ezir_ctx = IrrelevantManager(image_ids)
        # load QA data
        d = load_hdf5(data_file)
        self._r_arr = d['quest_arr'].astype(np.int32)
        self._r_arr_len = d['quest_len'].astype(np.int32)
        self._num_rel = len(self._r_images)
        # load blacklist
        blist = load_json('data/kptest_blacklist.json')
        is_valid = np.array([image_id not in blist for image_id in image_ids])
        print('%d/%d valid' % (is_valid.sum(), is_valid.size))
        self.valid_ids = np.where(is_valid)[0]

    def pop_rel_batch(self):
        index = np.random.choice(self.valid_ids, size=(self._rel_batch_size,),
                                 replace=False)
        feats = self._load_image_features(self._r_images, index)
        q, q_len = self._slice_questions(self._r_arr, self._r_arr_len, index)
        outputs = [index, feats, q, q_len]
        return outputs

    def pop_ez_ir_batch(self, rel_index, rel_feats):
        local_ids = np.random.choice(self._rel_batch_size,
                                     self._ez_ir_batch_size,
                                     replace=False)
        feats = rel_feats[local_ids, ::]
        index = self.ezir_ctx.query_irrelevant(rel_index[local_ids])  # questions from other images
        q, q_len = self._slice_questions(self._r_arr, self._r_arr_len, index)
        outputs = [feats, q, q_len]
        return outputs

    def pop_ir_batch(self):
        index = np.random.choice(self._num_ir, size=(self._ir_batch_size,),
                                 replace=False)
        feats = self._load_image_features(self._ir_images, index)
        q, q_len = self._slice_questions(self._ir_arr, self._ir_arr_len, index)
        outputs = [feats, q, q_len]
        return outputs

    def pop_batch(self):
        rel_index, imr, qr, qlr = self.pop_rel_batch()
        lbr = np.ones_like(qlr, dtype=np.int32)
        im_ez, qez, qlez = self.pop_ez_ir_batch(rel_index, imr)
        lbez = np.zeros_like(qlez, dtype=np.int32)
        imir, qir, qlir = self.pop_ir_batch()
        lbir = np.zeros_like(qlir, dtype=np.int32)
        # concat
        im = np.concatenate([imr, im_ez, imir], axis=0)
        quest = _concat_arrays(_concat_arrays(qr, qez), qir)
        quest_len = np.concatenate([qlr, qlez, qlir])
        labels = np.concatenate([lbr, lbez, lbir])
        return [im, quest, quest_len, labels]

    def get_next_batch(self):
        batch_data = self.pop_batch()
        self._queue.put(batch_data)

    def run(self):
        print('DataFetcher started')
        np.random.seed(self._proc_id)
        # very important, it ensures multiple processes didn't generate the same index
        while True:
            self.get_next_batch()

    def _load_image_features(self, images, index):
        feats = []
        for idx in index:
            filename = images[idx]
            try:
                f = np.load(os.path.join(self._FEAT_ROOT, filename + '.npz'))['x']
                if self._transpose_feat:
                    feats.append(f.transpose((1, 2, 0))[np.newaxis, ::])
                else:
                    feats.append(f[np.newaxis, ::])
            except Exception as e:
                print('Process %d: Error loading file: %s' % (self._proc_id,
                                                              os.path.join(self._FEAT_ROOT,
                                                                           filename + '.npz')))
                print(str(e))
                feats.append(np.zeros([1, 14, 14, 2048], dtype=np.float32))

        return np.concatenate(feats, axis=0).astype(np.float32)

    def _slice_questions(self, quest, quest_len, index):
        q_len = quest_len[index]
        max_len = q_len.max()
        seq = quest[index, :max_len]
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
        meta_file = os.path.join('data/QRD_irrelevant_meta_test.json')
        data_file = os.path.join('data/QRD_irrelevant_data_test.data')
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)
        # load QA data
        d = load_hdf5(data_file)
        self._quest = d['quest'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._labels = d['labels'].astype(np.float32)
        # self._check_valid_answers()
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
        # image_id = self._vqa_image_ids[index]
        feats = self._load_image_features(index)
        q, q_len = self._slice_questions(index)
        a = self._slice_answers(index)
        # quest_id = self._quest_ids[index]
        return feats, q, q_len, a

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
        return self._labels[index]


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
