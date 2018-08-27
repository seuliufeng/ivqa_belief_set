from post_process_variation_questions import post_process_variation_questions_noise, prepare_reinforce_data, \
    wrap_samples_for_language_model_v2
from experience_replay import ReplayBuffer

_replay_buffer = ReplayBuffer(batch_size=16, ratio=2)


def reinforce_trainstep(reader, model, env, sess, task_ops):
    outputs = reader.pop_batch()
    quest_ids, res5c, images, quest, quest_len, top_ans, ans, ans_len = outputs
    # random sampling
    noise_vec, pathes, scores = model.random_sampling([images, ans, ans_len], sess)
    _this_batch_size = images.shape[0]
    scores, pathes, noise = post_process_variation_questions_noise(scores,
                                                                   pathes,
                                                                   noise_vec,
                                                                   _this_batch_size,
                                                                   find_unique=False)

    wrapped_sampled, sampled_flat = \
        wrap_samples_for_language_model_v2(sampled=pathes,
                                           pad_token=model.pad_token - 1,
                                           max_length=20)
    # compute reward
    vqa_inputs = [images, res5c, ans, ans_len, top_ans]
    rewards, rewards_all, is_gt, aug_data = env.get_reward(pathes, [quest, quest_len],
                                                           [vqa_inputs, wrapped_sampled,
                                                            scores, quest_ids])

    max_path_arr, max_path_len, max_noise, max_rewards = \
        prepare_reinforce_data(pathes, noise, rewards, pad_token=model.pad_token)

    aug_images, aug_ans, aug_ans_len, is_in_vocab = aug_data
    sess_in = [aug_images, max_path_arr, max_path_len, aug_ans, aug_ans_len,
               max_noise, max_rewards, rewards_all]
    sess_in = [_in[is_in_vocab] for _in in sess_in]  # remove oov
    avg_reward = max_rewards.mean()

    # train op
    sess_outputs = sess.run(task_ops, feed_dict=model.fill_feed_dict(sess_in))
    sess_outputs += [avg_reward, 'reward']

    # update language model
    lm_scores = rewards_all[:, 2].flatten()
    env.lm.trainstep(_replay_buffer.get_batch())
    _replay_buffer.insert(sampled_flat, lm_scores)
    return sess_outputs
