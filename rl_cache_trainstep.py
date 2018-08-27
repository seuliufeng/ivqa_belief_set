from post_process_variation_questions import post_process_variation_questions_noise, prepare_reinforce_data, \
    wrap_samples_for_language_model, _parse_gt_questions, put_to_array, correct_language_model_inputs
import numpy as np
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


_Q_CTX = QuestionContext(batch_size=2 * 16, pad_token=15953)


def reinforce_trainstep(reader, model, env, sess, task_ops):
    outputs = reader.pop_batch()
    quest_ids, images, quest, quest_len, top_ans, ans, ans_len = outputs
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
    num_fake_in_batch = 80 - is_gt.sum()
    if num_fake_in_batch > 50 or True:  # at least half is generated
        wrapped_gt = _Q_CTX.get_gt_batch(*lm_inputs[2:])  # random sample new
        corrected_inputs = correct_language_model_inputs(wrapped_sampled + wrapped_gt, is_gt)
        # num_fake = corrected_inputs[0].shape[0]
        # num_real = corrected_inputs[2].shape[0]
        # print('Num positive: %d, num negative %d' % (num_real, num_fake))

        # _show_examples(corrected_inputs[0], corrected_inputs[1], np.zeros_like(corrected_inputs[1]), 'Fake')
        # _show_examples(corrected_inputs[2], corrected_inputs[3], np.zeros_like(corrected_inputs[3]), 'Real')
        # pdb.set_trace()
        if num_fake_in_batch > 0:
            env.lm.trainstep(corrected_inputs)
    return sess_outputs
