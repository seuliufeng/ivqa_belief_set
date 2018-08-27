import numpy as np
import os
from util import load_hdf5, load_json, get_feature_root, \
    find_image_id_from_fname, load_feature_file_vqabaseline
from multiprocessing import Process, Queue
# from readers.curriculum_sampler import CurriculumSampler

_DATA_ROOT = './'
data_root = 'data3/'


class Reader(object):
    def __init__(self, batch_size=32, subset='kptrain',
                 model_name='', feat_type='res5c',
                 version='v1', counter_sampling=False,
                 sample_negative=False,
                 use_fb_data=False):
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
        self._counter_sampling = counter_sampling
        self.sample_negative = sample_negative
        self.use_fb_data = use_fb_data

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
                                           counter_sampling=self._counter_sampling,
                                           sample_negative=self.sample_negative,
                                           use_fb_data=self.use_fb_data)
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
        data = self._data_queue.get()
        return data


class AttentionDataPrefetcher(Process):
    def __init__(self, output_queue, proc_id,
                 batch_size=32, subset='trainval',
                 feat_type='res5c', version_suffix='',
                 counter_sampling=False,
                 sample_negative=False,
                 use_fb_data=False):
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
        self.use_fb_data = use_fb_data
        self._sample_negative = sample_negative
        self._load_data()
        self._feat_type = feat_type.lower()
        self._FEAT_ROOT = get_feature_root(self._subset, self._feat_type)
        self._transpose_feat = 'res5c' in self._feat_type

    def _load_data(self):
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.meta' %
                                 (self._version_suffix, self._subset))
        data_file = os.path.join(_DATA_ROOT,
                                 'data3/%svqa_mc_w2v_coding_%s.data' %
                                 (self._version_suffix, self._subset))
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        self._quest_ids = np.array(d['quest_id'])
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)

        # load QA data
        d = load_hdf5(data_file)
        self._quests = d['quest_w2v'].astype(np.float32)
        self._answer = d['cands_w2v'].astype(np.float32)
        self._labels = d['labels']

        self._num = self._labels.size

        # double check question ids
        assert (np.all(self._quest_ids == d['quest_ids']))
        self._post_proc_answers()
        self._load_global_image_feature()

    def _post_proc_answers(self):
        if self._sample_negative:
            _all_labels = np.arange(18)
            neg_labels = []
            for label in self._labels:
                neg_label = np.setdiff1d(_all_labels, label)
                neg_labels.append(neg_label[np.newaxis, :])
            self._pos_labels = self._labels
            self._neg_labels = np.concatenate(neg_labels, axis=0)
            # print('%d - %d' % (self._neg_labels.max(), self._neg_labels.min()))
            self._answer = self._answer.reshape([self._num, 18, 300])

    def _load_global_image_feature(self):
        if self.use_fb_data:
            if 'train' in self._subset:
                data_file = 'data/imagenet_train_features.h5'
            else:
                data_file = 'data/imagenet_val_features.h5'
            d = load_feature_file_vqabaseline(data_file)
        else:
            data_file = 'data/res152_std_mscoco_%s.data' % self._subset
            d = load_hdf5(data_file)
        image_ids = d['image_ids']
        self._feat = d['features']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2feat_index = np.array(vqa_index2att_index, dtype=np.int32)

    def sample_answers(self, index):
        sample_ids = np.random.randint(low=0, high=18,
                                       size=(self._batch_size,))
        gt_labels = self._labels[index]
        labels = (sample_ids == gt_labels).astype(np.float32)
        answer_w2v = self._answer[index, sample_ids, :].astype(np.float32)
        return answer_w2v, labels

    def pop_batch(self):
        index = self._get_question_index()
        # feats = self._slice_attributes(index)
        feats = self._slice_image(index)
        quest_w2v = self._slice_questions(index)
        answer_w2v, labels = self.sample_answers(index)
        outputs = [feats, quest_w2v, answer_w2v, labels]
        return outputs

    def _get_question_index(self):
        index = np.random.choice(self._num, size=(self._batch_size,),
                                 replace=False)
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
        return self._quests[index]

    def _slice_answers(self, index):
        return self._answer[index]

    def _slice_image(self, index):
        # a = self._slice_attributes(index)
        f = self._slice_global_feature(index)
        return f

    def _slice_global_feature(self, index):
        feat_index = self._vqa_index2feat_index[index]
        return self._feat[feat_index, :]


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
        meta_file = os.path.join(_DATA_ROOT,
                                 'data/%svqa_std_mscoco_%s.meta' %
                                 (self._version_suffix, self._subset))
        data_file = os.path.join(_DATA_ROOT,
                                 'data3/%svqa_mc_w2v_coding_%s.data' %
                                 (self._version_suffix, self._subset))
        cand_file = os.path.join(_DATA_ROOT,
                                 'data3/%svqa_mc_cands_%s.meta' %
                                 (self._version_suffix, self._subset))
        # load meta
        d = load_json(meta_file)
        self._images = d['images']
        self._quest_ids = np.array(d['quest_id'])
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)

        # load QA data
        d = load_hdf5(data_file)
        self._quests = d['quest_w2v'].astype(np.float32)
        self._answer = d['cands_w2v'].astype(np.float32)
        self._labels = d['labels']

        # load candiates
        self._cand_ans = load_json(cand_file)

        self._num = self._labels.size

        # double check question ids
        assert (np.all(self._quest_ids == d['quest_ids']))

        self._load_global_image_feature()

    def _load_global_image_feature(self):
        if self.use_fb_data:
            if 'train' in self._subset:
                data_file = 'data/imagenet_train_features.h5'
            else:
                data_file = 'data/imagenet_val_features.h5'
            d = load_feature_file_vqabaseline(data_file)
        else:
            data_file = 'data/res152_std_mscoco_%s.data' % self._subset
            d = load_hdf5(data_file)
        image_ids = d['image_ids']
        self._feat = d['features']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2feat_index = np.array(vqa_index2att_index, dtype=np.int32)

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num - self._idx)
        index = self._index[self._idx:self._idx + this_batch_size]
        self._idx += this_batch_size
        return index

    def get_test_batch(self):
        index = self._get_sequencial_index()
        image_id = self._vqa_image_ids[index]
        quest_id = self._quest_ids[index]
        # slice data
        feats = self._slice_image(index)
        quest_w2v = self._slice_questions(index)
        answer_w2v = self._slice_answers(index)
        # labels = self._labels[index]
        cands = [self._cand_ans[idx] for idx in index]
        return feats, quest_w2v, answer_w2v, cands, quest_id, image_id

    # def pop_batch(self):
    #     index = self._get_sequencial_index()
    #     feats = self._load_image_features(index)
    #     q, q_len = self._slice_questions(index)
    #     a = self._slice_answers(index)
    #     return feats, q, q_len, a

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
        return self._quests[index]

    def _slice_answers(self, index):
        return self._answer[index]

    def _slice_image(self, index):
        return self._slice_global_feature(index)

    def _slice_global_feature(self, index):
        feat_index = self._vqa_index2feat_index[index]
        return self._feat[feat_index, :]


if __name__ == '__main__':
    from time import time
    from inference_utils.question_generator_util import SentenceGenerator

    to_sentence = SentenceGenerator(trainset='trainval')
    reader = AttentionDataPrefetcher(batch_size=4, subset='kptrain')
    for i in range(10):
        print(i)
        data = reader.pop_batch()
