import numpy as np
import numpy.random as nr
import os
from util import save_hdf5, load_hdf5


def _weighted_sampling(weights, batch_size):
    cdf = np.cumsum(weights).reshape([1, -1])
    p = nr.rand(batch_size, 1)
    idx = (cdf >= p).argmax(axis=1)
    return idx


def _standardize_data_1d(data, skip_zero=False, EPS=1e-12):
    smp_tab = (data != 0)
    if smp_tab.sum() > 0:
        mu = data[smp_tab].mean()
        sigma = np.std(data[smp_tab])
        if skip_zero:
            data[smp_tab] = (data[smp_tab] - mu) / (EPS + sigma)
        else:
            data = (data - mu) / (EPS + sigma)
    return data


def _softmax_1d(x, EPS=1e-12):
    x = np.exp(x)
    return x / (x.sum() + EPS)


def _compute_exploitation_ratio(sorted_index, vis_count, ratio=0.1):
    vis_index = np.where(vis_count > 0)[0]
    tot_visit = vis_count.sum()
    db_size = sorted_index.size
    r = db_size / float(tot_visit)
    vis_in_topk = np.intersect1d(sorted_index[:int(tot_visit * ratio)],
                                 vis_index)
    n_visit_in_topk = vis_count[vis_in_topk].sum()
    return (r * n_visit_in_topk / float(tot_visit))


class CurriculumSampler(object):
    def __init__(self, batch_size=32, num_samples=None,
                 cand_pool_size=1000, epsilon=0.5, suffix=''):
        self._batch_size = batch_size
        self._db_size = num_samples
        self._gamma = 0.0
        self._epsilon = epsilon
        self._cand_pool_size = cand_pool_size
        self._T_loss = 1.0
        self._T_visit = 4.0
        self._stat_iterval = 500
        self.iter = 0
        self._index = None
        self._num_visit_prev = None
        self._cache_file = '.cache/curr_sampler.%s.h5' % suffix
        self._init_buffer()

    def decay_epsiolon(self, decay_fn):
        pass

    def _init_buffer(self):
        self._index = np.arange(self._db_size)
        if os.path.exists(self._cache_file):
            print('Restore history state from file %s' % self._cache_file)
            d = load_hdf5(self._cache_file)
            self._loss = d['loss']
            self._num_visit = d['num_visit']
        else:
            print('No previous cache file found, create new one %s' % self._cache_file)
            self._loss = 100. * np.ones(self._db_size, dtype=np.float32)
            self._num_visit = np.ones(self._db_size, dtype=np.float32)
        self._num_visit_prev = self._num_visit.copy()

    def backup_statistics(self):
        save_hdf5(self._cache_file, {'loss': self._loss,
                                     'num_visit': self._num_visit})

    def set_valid_index(self, index):
        self._index = index

    def update_loss(self, index, losses):
        if index is None or losses is None:
            return
        self._loss[index] = self._loss[index] * self._gamma + (1 - self._gamma) * losses
        self._num_visit[index] += 1

    def sample_batch(self):
        # random sample a large batch
        act_idx = nr.choice(self._index, size=self._cand_pool_size, replace=False)
        # compute statistics of active samples
        weights = self._compute_statistics(act_idx)
        idx_in_act = _weighted_sampling(weights, self._batch_size)
        self.iter += 1

        if self.iter % self._stat_iterval == 0:
            self.print_exploration_exploitation()
            # self._backup_statistics()
        return act_idx[idx_in_act]

    def print_exploration_exploitation(self):
        num_visit = self._num_visit - self._num_visit_prev
        # compute exploration ratio
        visit_tab = num_visit > 0
        num_vis_points = (visit_tab).sum()
        num_tot_visit = num_visit.sum()
        # compute exploitation ratio
        # stat_loss = self._loss[self._index]
        stat_visit = num_visit[self._index]
        tot_visit = self._num_visit[self._index]
        epoch_id = tot_visit.sum() / float(tot_visit.size) - 1
        # sorted_index = (-stat_loss).argsort()
        # exploit_ratio_10 = _compute_exploitation_ratio(sorted_index, stat_visit, ratio=0.1)
        # exploit_ratio_30 = _compute_exploitation_ratio(sorted_index, stat_visit, ratio=0.3)
        # exploit_ratio_50 = _compute_exploitation_ratio(sorted_index, stat_visit, ratio=0.5)
        num_hist_visit = (self._num_visit[self._index] - 1 > 0).sum()
        mean_loss = self._loss[visit_tab].mean()
        print('\nEpoch idx: %0.2f' % epoch_id)
        print('Total coverage: %0.2f' % (num_hist_visit * 100. / stat_visit.size))
        print('Mean batch loss: %0.2f' % mean_loss)
        print('Num visited points: %d' % num_vis_points)
        print('Exploration ratio: %0.2f' % (num_vis_points * 100. / num_tot_visit))
        # print('Exploitation ratio 10: %0.2f' % (exploit_ratio_10 * 100.))
        # print('Exploitation ratio 30: %0.2f' % (exploit_ratio_30 * 100.))
        # print('Exploitation ratio 50: %0.2f' % (exploit_ratio_50 * 100.))
        print('\n')
        # backup num visit
        self._num_visit_prev = self._num_visit.copy()

    def _compute_statistics(self, act_idx):
        loss = _standardize_data_1d(self._loss[act_idx].copy(), skip_zero=False)
        num_visit = _standardize_data_1d(self._num_visit[act_idx].copy(), skip_zero=False)
        w_loss = _softmax_1d(loss * self._T_loss)  # use hard samples (exploration)
        w_visit = _softmax_1d(-num_visit * self._T_visit)  # promote rare visited (exploitation)
        w_total = self._epsilon * w_loss + (1 - self._epsilon) * w_visit
        w_total = w_total / w_total.sum()  # re-normalize
        return w_total


if __name__ == '__main__':
    sampler = CurriculumSampler(batch_size=32, num_samples=100000, cand_pool_size=1000, epsilon=0.8)
    # d = _standardize_data_1d(np.zeros(4), EPS=1e-12)
    num_batches = 1002
    for i in range(num_batches):
        idx = sampler.sample_batch()
        loss = nr.rand(32)
        sampler.update_loss(idx, loss)
