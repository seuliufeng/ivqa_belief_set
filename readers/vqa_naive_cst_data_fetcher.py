import numpy as np
from readers.vqa_naive_data_fetcher import AttentionDataReader as ReaderWorker
from vqa_replay_buffer_manager import VQABufferManager


def concat_arrays(arr1, arr2):
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


class ContrastiveDataReader(object):
    def __init__(self, batch_size=32, subset='kprestval',
                 cst_file='vqa_replay_buffer/vqa_replay.json',
                 mode='all'):
        self.batch_size = batch_size
        self.mode = mode
        assert(self.mode in ['cst', 'real', 'all'])
        self.worker = ReaderWorker(batch_size=self.batch_size,
                                   subset=subset,
                                   version='v1',
                                   use_quest_id=True)
        self.cst_worker = VQABufferManager(cst_file)

    def start(self):
        self.worker.start()

    def stop(self):
        self.worker.stop()

    def pop_batch(self):
        outs = self.worker.pop_batch()
        quest_ids, images, quest, quest_len, top_ans = outs
        mask, cst_quest, cst_quest_len = self.cst_worker.query(quest_ids)
        real_mask = np.ones_like(mask, dtype=np.float32)
        if self.mode == 'cst':
            return images, cst_quest, cst_quest_len, top_ans, mask
        elif self.mode == 'real':
            return images, quest, quest_len, top_ans, real_mask
        elif self.mode == 'all':
            # concat or replicate data
            images = np.tile(images, [2, 1])
            quest = concat_arrays(quest, cst_quest)
            quest_len = np.concatenate([quest_len, cst_quest_len])
            top_ans = np.tile(top_ans, [2])
            mask = np.concatenate([real_mask, mask])  # concat mask
            return images, quest, quest_len, top_ans, mask
        else:
            raise Exception('unknown mode!')
