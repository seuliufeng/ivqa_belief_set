from post_process_variation_questions import post_process_variation_questions_noise, prepare_reinforce_data, \
    wrap_samples_for_language_model, _parse_gt_questions, put_to_array, correct_language_model_inputs
import numpy as np
from var_ivqa_rewards import serialize_path
import os
import pdb


class QuestionContext(object):
    def __init__(self, batch_size, pad_token):
        self.batch_size = batch_size
        from util import load_hdf5
        data_file = 'data/vqa_std_mscoco_kprestval.data'
        d = load_hdf5(data_file)
        gts = _parse_gt_questions(d['quest_arr'], d['quest_len'])
        gts = [_g + [2] for _g in gts]
        self._quest, self._quest_len = put_to_array(gts, pad_token,
                                                    max_length=20)
        self.num = self._quest_len.size

    def get_random_questions(self):
        index = np.random.choice(self.num, size=(self.batch_size,),
                                 replace=False)
        return self._quest[index], self._quest_len[index]

    def get_gt_batch(self, cur_arr, cur_arr_len):
        other_arr, other_arr_len = self.get_random_questions()
        arr = self._concat_arrays(cur_arr, other_arr)
        assert (arr.shape[1] == 20)
        arr_len = np.concatenate([cur_arr_len, other_arr_len], axis=0)
        return [arr, arr_len]

    @staticmethod
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


_Q_CTX = QuestionContext(batch_size=4, pad_token=15953)


class VQABelief(object):
    def __init__(self):
        self.thresh = 0.5
        self.iter = 0
        self.memory = {}

    def insert(self, pathes, scores):
        for p, sc in zip(pathes, scores):
            path_key = serialize_path(p)
            if path_key in p or sc < self.thresh:
                continue
            self.memory[path_key] = sc
        self.iter += 1
        # print(self.iter)

    def should_terminate(self):
        do_term = (self.iter > 200 or self.size >= 20) \
                  and (self.size > 0 or self.iter > 400)
        return do_term

    @property
    def size(self):
        return len(self.memory)

    def clear(self):
        self.iter = 0
        self.memory = {}

    def show_belief(self, env, quest_id):
        # os.system('clear')
        print('Iter %d: Find %d unique questions for question %d' %
              (self.iter, len(self.memory), quest_id))
        for p in self.memory.keys():
            v = self.memory[p]
            tokens = [int(t) for t in p.split(' ')]
            sent = env.to_sentence.index_to_question(tokens)
            print('%s (%0.2f)' % (sent, v))
        mean_score = np.mean(self.memory.values())
        print('Mean VQA score: %0.2f' % mean_score)
        print('\n')


def reinforce_trainstep(reader_outputs, model, env, sess, task_ops, _VQA_Belief):
    # reader_outputs = reader.pop_batch()
    # quest_ids, images, quest, quest_len, top_ans, ans, ans_len = reader_outputs
    # select the first image
    # idx = 0
    #
    # def _reshape_array(v):
    #     if type(v) == np.ndarray:
    #         return v[np.newaxis, :]
    #     else:
    #         return np.reshape(v, (1,))
    #
    # selected = [_reshape_array(v[idx]) for v in reader_outputs]
    images, quest, quest_len, top_ans, ans, ans_len, quest_ids, image_ids = reader_outputs
    # random sampling
    noise_vec, pathes, scores = model.random_sampling([images, ans, ans_len], sess)
    _this_batch_size = images.shape[0]
    scores, pathes, noise = post_process_variation_questions_noise(scores,
                                                                   pathes,
                                                                   noise_vec,
                                                                   _this_batch_size,
                                                                   find_unique=False)

    lm_inputs = wrap_samples_for_language_model(sampled=pathes,
                                                pad_token=model.pad_token - 1,
                                                gts=[quest, quest_len],
                                                max_length=20)

    def _show_examples(arr, arr_len, _rewards, name):
        ps = _parse_gt_questions(arr, arr_len)
        print('\n%s:' % (name))
        for p, r in zip(ps, _rewards):
            if p[-1] == 2:
                p = p[:-1]
            sent = env.to_sentence.index_to_question(p)
            print('%s (%d)' % (sent, r))

    # compute reward
    vqa_inputs = [images, ans, ans_len, top_ans]
    # lm_inputs = lm_inputs[:2]
    wrapped_sampled = lm_inputs[:2]
    rewards, rewards_all, is_gt, aug_data = env.get_reward(pathes, [quest, quest_len],
                                                           [vqa_inputs, wrapped_sampled,
                                                            scores, quest_ids])

    max_path_arr, max_path_len, max_noise, max_rewards = \
        prepare_reinforce_data(pathes, noise, rewards, pad_token=model.pad_token)

    vqa_scores = rewards_all[:, 0]
    language_scores = rewards_all[:, 2]
    # scores = vqa_scores * (language_scores > 0.5)
    scores = vqa_scores * (language_scores > env.language_thresh)
    new_pathes = _parse_gt_questions(max_path_arr, max_path_len)
    _VQA_Belief.insert(new_pathes, scores)

    # _show_examples(max_path_arr, max_path_len, is_gt, 'Sampled')
    # pdb.set_trace()

    aug_images, aug_ans, aug_ans_len, is_in_vocab = aug_data
    sess_in = [aug_images, max_path_arr, max_path_len, aug_ans, aug_ans_len,
               max_noise, max_rewards, rewards_all]
    sess_in = [_in[is_in_vocab] for _in in sess_in]  # remove oov
    avg_reward = max_rewards.mean()

    # train op
    sess_outputs = sess.run(task_ops, feed_dict=model.fill_feed_dict(sess_in))
    sess_outputs += [avg_reward, 'reward']

    # update language model
    # print('Number GT: %d' % is_gt.sum())
    # num_fake_in_batch = 80 - is_gt.sum()
    if False:  # at least half is generated
        wrapped_gt = _Q_CTX.get_gt_batch(*lm_inputs[2:])  # random sample new
        corrected_inputs = correct_language_model_inputs(wrapped_sampled + wrapped_gt, is_gt)
        # num_fake = corrected_inputs[0].shape[0]
        # num_real = corrected_inputs[2].shape[0]
        # print('Num positive: %d, num negative %d' % (num_real, num_fake))

        # _show_examples(corrected_inputs[0], corrected_inputs[1], np.zeros_like(corrected_inputs[1]), 'Fake')
        # _show_examples(corrected_inputs[2], corrected_inputs[3], np.zeros_like(corrected_inputs[3]), 'Real')
        # pdb.set_trace()

        if min(wrapped_sampled[1].size, wrapped_gt[1].size) > 0:
            env.lm.trainstep(corrected_inputs)
    return sess_outputs
