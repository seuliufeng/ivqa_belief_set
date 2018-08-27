import numpy as np
import os
import re
from time import time
from util import load_json, save_json
from var_ivqa_rewards import serialize_path


class MemoryBuffer(object):
    def __init__(self):
        self.memory = {}

    def insert(self):
        pass


def _insert_item(d, key, value=None):
    if key not in d:
        d[key] = value
    return d


def _insert_pathes(d, ps):
    for (p, v) in ps:
        key = serialize_path(p)
        _insert_item(d, key, float(v))
    return d


class VQAReplayBuffer(object):
    def __init__(self, insert_thresh=0.8,
                 sv_dir='vqa_replay_buffer'):
        self.thresh = insert_thresh
        self.memory = {}
        self.sv_dir = sv_dir
        self.num_call = 0
        self.sv_dir = sv_dir
        self.sv_format = 'vqa_replay-%d.json'
        self.save_interval = 2500

    def restore(self):
        ckpts = os.listdir(self.sv_dir)
        if ckpts:
            iters = [int(re.findall('\d+', ckpt)[-1]) for ckpt in ckpts]
            idx = int(np.argmax(iters))
            ckpt_file = os.path.join(self.sv_dir, ckpts[idx])
            print('Restore VQA replay buffer from file %s' % ckpt_file)
            d = load_json(ckpt_file)
            self.num_call = d['num_call']
            self.memory = d['memory']

    def insert(self, quest_ids, questions, scores):
        assert (len(quest_ids) == len(questions))
        idx = 0
        for quest_id, ps in zip(quest_ids, questions):
            if quest_id < 0:  # skip padding
                idx += len(ps)  # be sure to change idx correspondingly
                continue
            quest_key = str(quest_id)
            if quest_key not in self.memory:
                self.memory[quest_key] = {}
            # insert paths to this key
            tmp = []
            for p in ps:
                sc = scores[idx]
                if sc > self.thresh:
                    tmp.append((p, sc))
                idx += 1
            _insert_pathes(self.memory[quest_key], tmp)
        assert(idx == scores.size)
        self.num_call += 1
        self.save()  # save buffer state if it is required

    def query(self, ids):
        pathes = []
        for _id in ids:
            pathes.append(self.memory[str(_id)].keys())
        return pathes

    def save(self):
        if self.num_call % self.save_interval == 0:
            print('Saving VQA replay buffers')
            t = time()
            sv_file = os.path.join(self.sv_dir, 'vqa_replay.json')
            save_json(sv_file, {'num_call': self.num_call,
                                'memory': self.memory})
            print('File %s saved to disk, total time: %0.2fs' % (sv_file, time() - t))
