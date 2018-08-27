from models.model_creater import get_model_creation_fn
import tensorflow as tf
import numpy as np
# from test_variation_ivqa_beam_search import post_process_prediction
from inference_utils.question_generator_util import SentenceGenerator
from readers.ivqa_reader_creater import create_reader
from post_process_variation_questions import post_process_variation_questions_noise, prepare_reinforce_data
from var_ivqa_rewards import IVQARewards, _parse_gt_questions, VQARewards
from time import time
import pdb

model_fn = get_model_creation_fn('VAQ-VarRL')
model = model_fn()

model.build()
sess = tf.Session()
print('Init model')
model.init_fn(sess)

to_sentence = SentenceGenerator(trainset='trainval')

env = IVQARewards()
# env1 = IVQARewards(subset='kprestval')
env1 = VQARewards(ckpt_file='model/kprestval_VQA-BaseNorm/model.ckpt-26000')

create_fn = create_reader('VAQ-Var', phase='train')
reader = create_fn(batch_size=100, subset='kpval',
                   version='v1')
reader.start()

import grammar_check

tool = grammar_check.LanguageTool('en-US')


def process_questions(sentences):
    def _is_valid(s):
        q = to_sentence.index_to_question(s).replace(" 's", "'s").capitalize()
        return False if tool.check(q) else True

    is_valid = [_is_valid(s) for s in sentences]
    return is_valid


for i in range(10):
    print('iter: %d/%d' % (i, 10))
    # outputs = reader.get_test_batch()
    outputs = reader.pop_batch()

    # inference
    # quest_ids, image_ids = outputs[-2:]
    images, quests, quest_len, ans, ans_len = outputs

    t = time()
    gts = _parse_gt_questions(quests, quest_len)
    is_valid = process_questions(gts)
    print(sum(is_valid))
    print('Batch %d, time %0.3fs' % (i, time()-t))
    continue

    noise_vec, pathes, scores = model.random_sampling([images, ans, ans_len], sess)
    # print('Total time for sampling %0.3f' % (time() - t))

    _this_batch_size = images.shape[0]
    scores, pathes, noise = post_process_variation_questions_noise(scores,
                                                                   pathes,
                                                                   noise_vec,
                                                                   _this_batch_size)
    rewards = env.get_reward(pathes, [quests, quest_len])
    rewards1 = env1.get_reward(pathes, [images, ans, ans_len])
    # pdb.set_trace()

    max_path_arr, max_path_len, max_noise, max_rewards = \
        prepare_reinforce_data(pathes, noise, rewards, pad_token=-1)
    # print('Total time for RL data processing %0.3f' % (time()-t))
    print(max_path_arr)

    idx = 0
    gts = _parse_gt_questions(quests, quest_len)
    for _var_s, _var_n, _gt in zip(pathes, noise, gts):
        sentence = to_sentence.index_to_question(_gt)
        print('\nGT: %s' % sentence)
        _n = len(_var_s)
        ind = np.arange(idx, idx + _n, 1)
        _max_reward_idx = rewards[ind].argmax()
        _max_p = _var_s[_max_reward_idx]
        _max_n = _var_n[_max_reward_idx]
        sentence = to_sentence.index_to_question(_max_p)
        _max_r = rewards[ind][_max_reward_idx]
        _max_r1 = rewards1[ind][_max_reward_idx]
        print('%s (%0.3f, %0.3f)' % (sentence, _max_r, _max_r1))
        idx += _n



        # pdb.set_trace()
        #
        # print('\n')

        # idx = 0
        # for _var_s, _gt in zip(pathes, gts):
        #     sentence = to_sentence.index_to_question(_gt)
        #     print('\nGT: %s' % sentence)
        #     for _s in _var_s:
        #         sentence = to_sentence.index_to_question(_s)
        #         print('%s (%0.3f)' % (sentence, rewards[idx]))
        #         idx += 1

reader.stop()
