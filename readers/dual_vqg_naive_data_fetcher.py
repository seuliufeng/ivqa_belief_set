import numpy as np
from readers.vqa_naive_data_fetcher import AttentionDataReader as ReaderWorker
from readers.vqa_naive_vqg_flt_cand_data_fetcher import AttentionDataReader as ReaderWorkerFlt


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


class DualReader(object):
    def __init__(self, batch_size=32,
                 known_set='kprestval',
                 unknown_set='kptrain',
                 un_ratio=1,
                 hide_label=True):
        self.hide_label = hide_label
        kn_batch_size = int(batch_size / (un_ratio + 1))
        self.kn_batch_size = kn_batch_size
        un_batch_size = batch_size - kn_batch_size
        print('Semi Data Reader:')
        print('Batch size: known: %d, unknown: %d' % (kn_batch_size, un_batch_size))
        self.kn_worker = ReaderWorker(batch_size=kn_batch_size,
                                      subset=known_set,
                                      version='v1',
                                      n_process=1)
        self.use_un_worker = un_batch_size != 0
        if self.use_un_worker:
            self.un_worker = ReaderWorkerFlt(batch_size=un_batch_size,
                                             subset=unknown_set,
                                             version='v1')

    def start(self):
        self.kn_worker.start()
        if self.use_un_worker:
            self.un_worker.start()

    def stop(self):
        self.kn_worker.stop()
        if self.use_un_worker:
            self.un_worker.stop()

    def pop_batch(self):
        kn_outs = self.kn_worker.pop_batch()
        if self.use_un_worker:
            un_outs = self.un_worker.pop_batch()
            # Concat batch
            kn_im, kn_q, kn_q_len, kn_a = kn_outs
            un_im, un_q, un_q_len, un_a = un_outs
            im = np.concatenate([kn_im, un_im], axis=0)
            q = _concat_arrays(kn_q, un_q)
            q_len = np.concatenate([kn_q_len, un_q_len], axis=0)
            a = np.concatenate([kn_a, un_a], axis=0)
            return [im, q, q_len, a]
        else:
            return kn_outs

    def pop_labeled(self):
        return self.kn_worker.pop_batch()

    def pop_unlabeled(self):
        return self.un_worker.pop_batch()

    def mix_batch(self, un_outs):
        kn_outs = self.kn_worker.pop_batch()
        # Concat batch
        kn_im, kn_q, kn_q_len, kn_a = kn_outs
        un_im, un_q, un_q_len, un_a = un_outs
        im = np.concatenate([kn_im, un_im], axis=0)
        q = _concat_arrays(kn_q, un_q)
        q_len = np.concatenate([kn_q_len, un_q_len], axis=0)
        a = np.concatenate([kn_a, un_a], axis=0)
        mask = np.ones_like(a, dtype=np.float32)
        mask[self.kn_batch_size:] = 0.
        return [im, q, q_len, a, mask]
