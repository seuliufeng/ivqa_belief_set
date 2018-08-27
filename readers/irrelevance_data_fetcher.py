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
        # self._rel_batch_size = int(self._batch_size * 0.5)
        self._ez_ir_batch_size = int(self._batch_size * 0.6)
        self._ir_batch_size = self._batch_size - self._ez_ir_batch_size

    def _load_data(self):
        self._load_irrelevant_data()
        self._load_relevant_data()

    def _load_irrelevant_data(self):
        # load meta
        d = load_json('/data1/fl302/projects/inverse_vqa/data/QRD_irrelevant_meta.json')
        self._ir_images = d['images']
        # load QA data
        d = load_hdf5('/data1/fl302/projects/inverse_vqa/data/QRD_irrelevant_data.data')
        self._ir_arr = d['quest'].astype(np.int32)
        self._ir_arr_len = d['quest_len'].astype(np.int32)
        self._num_ir = len(self._ir_images)

    def _load_relevant_data(self):
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/v2_vqa_std_mscoco_%s.meta' % self._subset)
        data_file = os.path.join(_DATA_ROOT,
                                 'data/v2_vqa_std_mscoco_%s.data' % self._subset)
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

    def pop_ez_ir_batch(self, rel_index):
        index = self.ezir_ctx.query_irrelevant(rel_index)  # questions from other images
        feats = self._load_image_features(self._r_images, index)
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
        # rel_index, imr, qr, qlr = self.pop_rel_batch()
        # lbr = np.ones_like(qlr, dtype=np.int32)
        index = np.random.choice(self.valid_ids,
                                 size=(self._ez_ir_batch_size,),
                                 replace=False)
        im_ez, qez, qlez = self.pop_ez_ir_batch(index)
        # lbez = np.zeros_like(qlez, dtype=np.int32)
        imir, qir, qlir = self.pop_ir_batch()
        # lbir = np.zeros_like(qlir, dtype=np.int32)
        # concat
        im = np.concatenate([im_ez, imir], axis=0)
        quest = _concat_arrays(qez, qir)
        quest_len = np.concatenate([qlez, qlir])
        return [im, quest, quest_len]

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
