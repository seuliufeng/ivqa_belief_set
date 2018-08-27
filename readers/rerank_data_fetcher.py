from __future__ import division
import numpy as np
import os
from util import load_hdf5, load_json
from multiprocessing import Process, Queue
from math import ceil

# FEAT_ROOT = 'data/resnet_res5c'
FEAT_ROOT = 'data'


# FEAT_ROOT = '/import/vision-ephemeral/fl302/code/text-to-image'


def _cat_sequences(seq_arr1, seq_arr2):
    max_len_1 = seq_arr1.shape[1]
    max_len_2 = seq_arr2.shape[1]
    max_len = max(max_len_1, max_len_2)
    n_pad_1 = max_len - max_len_1
    n_pad_2 = max_len - max_len_2
    seq_arr1 = np.pad(seq_arr1, ((0, 0), (0, n_pad_1)),
                      mode='constant', constant_values=0)
    seq_arr2 = np.pad(seq_arr2, ((0, 0), (0, n_pad_2)),
                      mode='constant', constant_values=0)
    return np.concatenate([seq_arr1, seq_arr2], axis=0)


def _create_sample_indicator_mask(num_tot_pairs, n_pairs_per_im=3):
    num_images = num_tot_pairs // n_pairs_per_im
    n_flip_per_im = 1 + np.random.randint(low=0, high=n_pairs_per_im,
                                          size=num_images)
    ORDER = np.random.rand(num_images, n_pairs_per_im).argsort(axis=1)
    flip_tab = np.zeros_like(ORDER, dtype=np.bool)
    for w, nw, ind in zip(flip_tab, n_flip_per_im, ORDER):
        w[ind[:nw]] = True
    return flip_tab.flatten()


class RetrievalDataReader(object):
    def __init__(self, batch_size=32, n_contrast=15, subset='train'):
        self._data_queue = None
        self._prefetch_procs = []
        self._batch_size = batch_size
        self._subset = subset
        self._n_process = 1
        self._n_contrast = n_contrast

    def start(self):
        self._data_queue = Queue(3)
        for proc_id in range(self._n_process):
            proc = RetrievalDataPrefetcher(self._data_queue,
                                           proc_id,
                                           self._batch_size,
                                           n_contrast=self._n_contrast,
                                           subset=self._subset)
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


# class RetrievalDataPrefetcher(object):
class RetrievalDataPrefetcher(Process):
    def __init__(self, queue=None, proc_id=123, batch_size=32,
                 n_contrast=15, subset='train', test_mode=False):
        super(RetrievalDataPrefetcher, self).__init__()
        self._batch_size = batch_size
        self._proc_id = proc_id
        self._n_contrast = n_contrast
        self._hard_qa_frac = 5
        self._easy_qa_frac = self._n_contrast - self._hard_qa_frac
        self._is_test = test_mode
        self._n_cst_in_batch = self._batch_size * self._n_contrast
        self._queue = queue
        self._subset = subset
        self._n_qa_per_im = 3
        self._n_cands = 17
        self._num = None
        self._image_index2qa_index = None
        self._use_wrong_answer = True
        self._n_cst_qa_in_batch = self._batch_size * (self._n_contrast -
                                                      self._use_wrong_answer)
        self._im_feats = None
        self._load_data()
        self._cand_index = np.arange(self._num)

        if self._is_test:
            self._num_batches = int(ceil(self._num /
                                         float(batch_size)))
            self._idx = 0
            self._index = np.arange(self._num)

    def _load_data(self):
        self._load_image_data()
        self._load_question_data()
        self._load_contrastive_answers()
        self._merge_answers()
        self._build_im2qa_index()

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num - self._idx)
        index = self._index[self._idx:self._idx + this_batch_size]
        self._idx += this_batch_size
        qa_index = self._to_qa_matrix_index(index)
        return index, qa_index, qa_index

    def _get_training_index_(self):
        index_pos = np.random.choice(self._num,
                                     size=self._batch_size,
                                     replace=False)
        _ind_pool = np.setdiff1d(self._cand_index, index_pos)
        neg_im_index = np.random.choice(_ind_pool,
                                        size=self._n_cst_in_batch,
                                        replace=False)
        neg_qa_index = np.random.choice(_ind_pool,
                                        size=self._n_cst_in_batch,
                                        replace=False)
        im_index = np.concatenate([index_pos, neg_im_index])
        qa_im_index = np.concatenate([index_pos, neg_qa_index])
        qa_index = self._to_qa_matrix_index(qa_im_index)
        return im_index, qa_index

    def _get_training_index(self):
        index_pos = np.random.choice(self._num,
                                     size=self._batch_size,
                                     replace=False)
        _ind_pool = np.setdiff1d(self._cand_index, index_pos)
        neg_im_index = np.random.choice(_ind_pool,
                                        size=self._n_cst_in_batch,
                                        replace=False)
        im_index = np.concatenate([index_pos, neg_im_index])
        pos_qa_index = self._to_qa_matrix_index(index_pos)
        cst_q_index, cst_a_index = self._sample_contrast_qa_pairs(index_pos)
        q_index = np.concatenate([pos_qa_index] + cst_q_index)
        a_index = np.concatenate([pos_qa_index] + cst_a_index)
        return im_index, q_index, a_index

    def _sample_contrast_qa_pairs(self, pos_im_index):
        _ind_pool = np.setdiff1d(self._cand_index, pos_im_index)
        quest_index, answer_index = [], []

        # true question, wrong answer
        for i in range(self._hard_qa_frac):
            q_ind, cands_batch = self._sample_quest_and_cst_answer(pos_im_index)
            a_ind = [cands[np.random.randint(self._n_cands)] for cands in cands_batch]

            hard_q_ind = np.array(q_ind, dtype=np.int32)[:, np.newaxis]
            hard_a_ind = np.array(a_ind, dtype=np.int32)[:, np.newaxis]
            quest_index.append(hard_q_ind)
            answer_index.append(hard_a_ind)

        # wrong answer, wrong question
        n_easy_qas = self._easy_qa_frac * self._batch_size
        neg_im_index = np.random.choice(_ind_pool,
                                        size=n_easy_qas,
                                        replace=False)
        cst_qa_index = self._to_qa_matrix_index(neg_im_index)
        quest_index.append(cst_qa_index.reshape([self._batch_size, self._easy_qa_frac]))
        answer_index.append(cst_qa_index.reshape([self._batch_size, self._easy_qa_frac]))
        quest_index = np.concatenate(quest_index, axis=1).flatten(order='C')
        answer_index = np.concatenate(answer_index, axis=1).flatten(order='C')
        return [quest_index], [answer_index]

    def _load_question_data(self):
        data_file = 'data/vqa_retrieval_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._quest_image_ids = d['image_ids'].astype(np.int32)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['ans_arr'].astype(np.int32)
        self._answer_len = d['ans_len'].astype(np.int32)
        self._quest_ids = d['quest_ids'].astype(np.int32)

    def _load_image_data(self):
        feat_file = os.path.join(FEAT_ROOT, 'mscoco_res152_%s.h5' % self._subset)
        d = load_hdf5(feat_file)
        self._im_feats = d['features']
        self._feats_image_ids = d['image_ids']
        self._num = self._im_feats.shape[0]

    def _load_contrastive_answers(self):
        if self._is_test:
            return
        meta_file = 'data/vqa_retrieval_cst_ans_mscoco_%s.meta' % self._subset
        data_file = 'data/vqa_retrieval_cst_ans_mscoco_%s.data' % self._subset
        d = load_json(meta_file)['quest_id2cand_index']
        self.quest_id2cand_index = {int(k): v for k, v in d.iteritems()}
        d = load_hdf5(data_file)
        self.cst_answers = d['cand_arr'].astype(np.int32)
        self.cst_ans_len = d['cand_len'].astype(np.int32)

    def _merge_answers(self):
        if self._is_test:
            return
        num = self._answer_len.size
        # offset candidate answers
        self.quest_id2cand_index = {k: [vv + num for vv in v] for k, v in
                                    self.quest_id2cand_index.iteritems()}
        # concat answer with contrastive answers
        self._answer = _cat_sequences(self._answer, self.cst_answers)
        self._answer_len = np.concatenate([self._answer_len, self.cst_ans_len])

    def _build_im2qa_index(self):
        # merge qa position (index) to image id
        image_id2quest_index = {}
        image_id2cand_index = {}
        for i, image_id in enumerate(self._quest_image_ids):
            if image_id in image_id2quest_index:
                image_id2quest_index[image_id].append(i)
                quest_id = self._quest_ids[i]
                if not self._is_test:
                    image_id2cand_index[image_id].append(self.quest_id2cand_index[quest_id])
            else:
                image_id2quest_index[image_id] = [i]
                quest_id = self._quest_ids[i]
                if not self._is_test:
                    image_id2cand_index[image_id] = [self.quest_id2cand_index[quest_id]]
        # change to the key to image position (index not id)
        self._image_index2qa_index = {}
        self._image_index2cand_index = {}
        for i, image_id in enumerate(self._feats_image_ids):
            self._image_index2qa_index[i] = image_id2quest_index[image_id]
            if not self._is_test:
                self._image_index2cand_index[i] = image_id2cand_index[image_id]

    def pop_batch(self):
        if self._is_test:
            im_index, q_index, a_index = self._get_sequencial_index()
        else:
            im_index, q_index, a_index = self._get_training_index()
        im = self._slice_images(im_index)
        q, q_len = self._slice_questions(q_index)
        a, a_len = self._slice_answers(a_index)
        return im, q, q_len, a, a_len

    def get_next_batch(self):
        output = self.pop_batch()
        self._queue.put(output)

    def _to_qa_matrix_index(self, index):
        return np.array([self._image_index2qa_index[idx][np.random.randint(self._n_qa_per_im)] for idx in index])

    def _sample_quest_and_cst_answer(self, index):
        quest_ind, cand_ind = [], []
        for idx in index:
            qa_ids = self._image_index2qa_index[idx]
            cand_ids = self._image_index2cand_index[idx]
            i = np.random.randint(self._n_qa_per_im)
            quest_ind.append(qa_ids[i])
            cand_ind.append(cand_ids[i])
        return quest_ind, cand_ind

    def run(self):
        print('DataFetcher started')
        np.random.seed(self._proc_id)
        # very important, it prevent multiple processes generating the same data
        while True:
            self.get_next_batch()

    def _slice_images(self, index):
        im = self._im_feats[index]
        return im.astype(np.float32)

    def _slice_questions(self, index):
        q_len = self._quest_len[index]
        max_len = q_len.max()
        seq = self._quest[index, :max_len]
        return seq, q_len

    def _slice_answers(self, index):
        a_len = self._answer_len[index]
        max_len = a_len.max()
        seq = self._answer[index, :max_len]
        return seq, a_len


class RerankTestFetcher(object):
    def __init__(self, batch_size=32):
        self._batch_size = batch_size
        self._idx = 0
        self._n_cands = 10
        self.load_data()
        self._num = self.quest_ids.size
        self._index = np.arange(self._num)

    def load_data(self):
        self._load_qa_data()
        self._load_image_data()

    def _load_qa_data(self):
        data_file = 'data/att_rerank/qa_arr.h5'
        d = load_hdf5(data_file)
        self._quest_arr = d['quest_arr']
        self._quest_len = d['quest_len']
        self._top2000_arr = d['top2000_arr']
        self._top2000_len = d['top2000_len']
        self._model_topk_pred = d['att_cand_arr']
        self.quest_ids = d['quest_id']
        self._image_ids = d['image_id']

    def _load_image_data(self):
        feat_file = os.path.join(FEAT_ROOT, 'mscoco_res152_dev_full.h5')
        d = load_hdf5(feat_file)
        self._im_feats = d['features']
        self._feats_image_ids = d['image_ids']
        self._num = self._im_feats.shape[0]
        self._image_id2image_index = {im_id: i for i, im_id in enumerate(self._feats_image_ids)}

    def _get_qa_sequences(self, index):
        a_index = self._model_topk_pred[index, :].flatten()
        q_arr, q_len = self._quest_arr[index, :], self._quest_len[index]
        qdim = q_arr.shape[-1]

        q_arr = np.tile(q_arr[:, np.newaxis, :], [1, self._n_cands, 1])
        q_arr = q_arr.reshape([-1, qdim])
        q_len = np.tile(q_len[:, np.newaxis], [1, self._n_cands]).flatten()
        return q_arr, q_len, self._top2000_arr[a_index], self._top2000_len[a_index]

    def _get_image_feature(self, index):
        im_index = [self._image_id2image_index[idx] for idx in self._image_ids[index]]
        image_feats = self._im_feats[im_index, :]
        return image_feats

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num - self._idx)
        index = self._index[self._idx:self._idx + this_batch_size]
        self._idx += this_batch_size
        return index

    def pop_batch(self):
        index = self._get_sequencial_index()
        im = self._get_image_feature(index)
        q_arr, q_len, a_arr, a_len = self._get_qa_sequences(index)
        return im, q_arr, q_len, a_arr, a_len


def test_rerank_reader():
    reader = RetrievalDataReader(batch_size=1, n_contrast=10, subset='train')
    reader.start()
    outputs = reader.pop_batch()
    im_feat, quest_arr, quest_len, ans_arr, ans_len = outputs
    from inference_utils.question_generator_util import SentenceGenerator
    to_sentence = SentenceGenerator(trainset='trainval',
                                    ans_vocab_file='data/vqa_trainval_question_answer_word_counts.txt',
                                    quest_vocab_file='data/vqa_trainval_question_answer_word_counts.txt')
    for q_seq, q_len, a_seq, a_len in zip(quest_arr, quest_len, ans_arr, ans_len):
        q_ = np.array([0] + q_seq[:q_len].tolist() + [0])
        a_ = np.array([0] + a_seq[:a_len].tolist() + [0])
        q = to_sentence.index_to_question(q_)
        a = to_sentence.index_to_answer(a_)
        print('Q: %s' % q)
        print('A: %s\n' % a)
    reader.stop()


if __name__ == '__main__':
    from time import time

    test_rerank_reader()
    exit(0)


    def test_test_reader():
        t = time()
        reader = RerankTestFetcher(batch_size=100)
        for i in range(130):
            data = reader.pop_batch()
            print(data[0].shape)
            print(data[1].shape)
            print(data[3].shape)
            data[0].mean()
            print(data[1].mean())
            data[2].max()

        avg_time = (time() - t) / 100.
        print('run 100 batches, avg time: %0.2f sec/batch' % avg_time)


    def test_train_reader():
        reader = RetrievalDataPrefetcher(None, 1, batch_size=32, n_contrast=15, subset='train', test_mode=False)
        t = time()

        for i in range(100):
            data = reader.pop_batch()
            print(data[0].shape)
            print(data[1].shape)
            print(data[3].shape)
            data[0].mean()
            print(data[1].mean())
            data[2].max()

        avg_time = (time() - t) / 100.
        print('run 100 batches, avg time: %0.2f sec/batch' % avg_time)


    test_test_reader()
