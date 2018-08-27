import os
from time import time, sleep
import tensorflow as tf
import json
from collections import OrderedDict
from shutil import copyfile
import numpy as np
from util import get_model_iteration

_V2_SUFFIX = ['.data-00000-of-00001', '.index', '.meta']


def mkdir_if_missing(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_model(src, dst):
    if os.path.exists(src):  # v1
        copyfile(src, dst)
    else:  # v2
        try:
            for suf in _V2_SUFFIX:
                copyfile(src + suf, dst + suf)
        except Exception, e:
            print('Model vanished')
            return


def delete_model(model_path):
    if os.path.exists(model_path):  # v1
        os.remove(model_path)
    else:  # v2
        for suf in _V2_SUFFIX:
            os.remove(model_path + suf)


def remove_model_suffix(model_path):
    for suf in _V2_SUFFIX:
        if suf in model_path:
            return os.path.splitext(model_path)[0]
    return model_path


class ModelWatcher(object):
    def __init__(self, model_dir, eval_fn, save_model=True,
                 n_keep=2, wait_sec=20, dir_map=None):
        self._model_dir = model_dir
        self._n_keep = n_keep
        self._wait_sec = wait_sec
        self._model_res_dict = {}
        self._prev_model = None
        self._eval_fn = eval_fn
        self._save_model = save_model
        self._res_dict = OrderedDict()
        self._dir_map = dir_map
        self._res_file = os.path.join('result/res_%s.json' % self.get_experiment_name())
        print(self._res_file)
        self._model_sv_dir = os.path.join(model_dir, '%s_best%d' % (self.get_experiment_name(),
                                                                    self._n_keep))
        self._load_result_file()
        mkdir_if_missing(self._model_sv_dir)

    def _load_result_file(self):
        if os.path.exists(self._res_file):
            print('Loading from previous results...')
            self._res_dict = json.load(open(self._res_file, 'r'))
            for model in self._res_dict:
                dst_file = os.path.join(self._model_sv_dir, model)
                acc = float(self._res_dict[model])
                self._model_res_dict[dst_file] = acc
            self._prev_model = self._res_dict.keys()[-1]
            self.print_results()

    def _vertify_model_dir(self, model_path):
        if self._dir_map is not None:
            _, md_file = os.path.split(model_path)
            return os.path.join(self._dir_map, md_file)
        else:
            return model_path

    def _check_for_new_model(self):
        ckpt = tf.train.get_checkpoint_state(self._model_dir)
        model_path = ckpt.model_checkpoint_path
        model_path = self._vertify_model_dir(model_path)
        if model_path == self._prev_model:
            print('Waiting for new models')
            sleep(self._wait_sec)
            return None
        else:
            self._prev_model = model_path
            print('Add new model %s to work list, evaluation starts in 30s' % os.path.basename(model_path))
            sleep(10)
            return model_path

    def get_experiment_name(self):
        return os.path.split(self._model_dir)[1]

    def write_result_to_file(self):
        json.dump(self._res_dict, open(self._res_file, 'w'))

    def print_results(self):
        # sort by iteration
        tmp = OrderedDict(sorted(self._res_dict.items(), key=lambda n: get_model_iteration(n[0])))
        for model in tmp.keys():
            print '%s: %s' % (model, tmp[model])

    def _backup_model(self, model_path):
        model_name = os.path.basename(model_path)
        dst_file = os.path.join(self._model_sv_dir, model_name)
        print('Backing up model %s' % model_name)
        copy_model(model_path, dst_file)
        acc = float(self._res_dict[model_name])
        self._model_res_dict[dst_file] = acc
        self._remove_bad_models()

    def _remove_bad_models(self):
        accs = np.array(self._model_res_dict.values())
        this_keep = min(self._n_keep, len(accs))
        thresh = np.sort(accs)[-this_keep]
        for model in os.listdir(self._model_sv_dir):
            model_path = os.path.join(self._model_sv_dir, model)
            if self._model_res_dict[remove_model_suffix(model_path)] < thresh:
                print('Removing model %s' % model_path)
                delete_model(model_path)

    def run(self):
        while True:
            model_path = self._check_for_new_model()
            if model_path is not None:
                try:
                    res_str = self._eval_fn(model_path)
                except Exception, e:
                    if e is KeyboardInterrupt:
                        raise
                    else:
                        print(str(e))
                        continue
                if type(res_str) == float:
                    res_str = '%0.3f' % res_str
                else:
                    assert (self._save_model == False)
                model_name = os.path.basename(model_path)
                self._res_dict[model_name] = res_str
                self.write_result_to_file()
                self.print_results()
                self._backup_model(model_path)


if __name__ == '__main__':
    from util import get_model_iteration
    from collections import OrderedDict
    import pylab as plt

    d = json.load(open('result/res_model.json'))
    # iters = [get_model_iteration(name) for name in d.keys()]
    new_d = OrderedDict(sorted(d.items(), key=lambda n: get_model_iteration(n[0])))
    print(new_d)
    x_axis = np.array([get_model_iteration(n) for n in new_d.keys()])
    y_axis = np.array(new_d.values())
    plt.plot(x_axis, y_axis)
    plt.grid()
    plt.show()
