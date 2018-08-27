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
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.meta' %
                                 (self._version_suffix, self._subset))
        data_file = os.path.join(_DATA_ROOT,
                                 'data4/%svar_ivqa_%s_question_answers.data' %
                                 (self._version_suffix, self._subset))
        # load meta
        d = load_json(meta_file)
        images = d['images']
        quest_ids = np.array(d['quest_id'])
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in images]
        quest_id2image_id = {qid: im_id for (qid, im_id) in zip(quest_ids, vqa_image_ids)}
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)

        # load QA data
        d = load_hdf5(data_file)
        self._quest_ids = d['ext_quest_ids'].astype(np.int32)
        self._quest = d['ext_quest_arr'].astype(np.int32)
        self._quest_len = d['ext_quest_len'].astype(np.int32)
        self._answer = d['ext_top_answer'].astype(np.int32)
        self._check_valid_answers()

        # sort images
        abs_quest_ids = self._quest_ids[:, 0]
        self._vqa_image_ids = np.array([quest_id2image_id[_id] for _id in abs_quest_ids],
                                       dtype=np.int32)

        # self._load_caption_feature()
        self._load_global_image_feature()

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2feat_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._feat = d['features']

    def _load_caption_feature(self):
        data_file = 'data/capt1k_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2att_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._attributes = d['att_arr']

    def _check_valid_answers(self):
        self._valid_ids = np.where(self._answer < self._num_top_ans)[0]
        print('%d/%d' % (self._valid_ids.size, self._answer.size))

    def pop_batch(self):
        index = np.random.choice(self._valid_ids, size=(self._batch_size,),
                                 replace=False)
        # feats = self._slice_attributes(index)
        feats = self._slice_image(index)
        q, q_len = self._slice_questions(index)
        a = self._slice_answers(index)
        outputs = [feats, q, q_len, a]
        return outputs

    def _get_question_index(self):
        index = self._get_question_index()
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

    def _slice_image(self, index):
        # a = self._slice_attributes(index)
        f = self._slice_global_feature(index)
        return f
        # return np.concatenate([a, f], axis=1)

    def _slice_attributes(self, index):
        attr_index = self._vqa_index2att_index[index]
        return self._attributes[attr_index, :]

    def _slice_global_feature(self, index):
        feat_index = self._vqa_index2feat_index[index]
        return self._feat[feat_index, :]


class AttentionFetcher(object):
    def __init__(self, batch_size, subset='kptest',
                 feat_type='res5c', version='v2',
                 var_suffix='var_'):
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
        self._var_suffix = var_suffix
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
        data_file = os.path.join(_DATA_ROOT,
                                 'data4/%svqg_%s_question_tokens.data' %
                                 (self._var_suffix, self._subset))
        # load QA data
        d = load_hdf5(data_file)
        self._quest_ids = d['ext_quest_ids'].astype(np.int32)
        self._quest = d['ext_quest_arr'].astype(np.int32)
        self._quest_len = d['ext_quest_len'].astype(np.int32)
        # self._answer = d['ext_top_answer'].astype(np.int32)
        self._vqa_image_ids = self._quest_ids[:, 0]
        # self._check_valid_answers()

        # sort images
        self._num = self._vqa_image_ids.size

        # self._load_caption_feature()
        self._load_global_image_feature()

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2feat_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._feat = d['features']

    def _load_caption_feature(self):
        data_file = 'data/capt1k_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2att_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._attributes = d['att_arr']

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
        feats = self._slice_image(index)
        q, q_len = self._slice_questions(index)
        # a = self._slice_answers(index)
        quest_id = self._quest_ids[index]
        return feats, q, q_len, quest_id, image_id

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

    def _slice_image(self, index):
        # a = self._slice_attributes(index)
        return self._slice_global_feature(index)
        # return np.concatenate([a, f], axis=1)

    def _slice_attributes(self, index):
        attr_index = self._vqa_index2att_index[index]
        return self._attributes[attr_index, :]

    def _slice_global_feature(self, index):
        feat_index = self._vqa_index2feat_index[index]
        return self._feat[feat_index, :]


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
