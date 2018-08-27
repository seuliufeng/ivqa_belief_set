import tensorflow as tf
import os
import time
import numpy as np
from config import VOCAB_CONFIG
import pdb
from inference_utils.question_generator_util import SentenceGenerator


def post_process_prediction_default(pathes):
    max_len = pathes.shape[1]
    pred_len = np.argmax(np.equal(pathes, VOCAB_CONFIG.end_token_id), axis=1)
    vaild_tab = pathes[:, 0] != VOCAB_CONFIG.end_token_id
    pred_len[np.logical_and(pred_len == 0, vaild_tab)] = max_len - 1
    confs, vpathes = [], []
    for path, plen in zip(pathes, pred_len):
        seq = path.tolist()[:plen+1]
        seq = [VOCAB_CONFIG.start_token_id] + seq
        vpathes.append(seq)
    return vpathes


def put_to_array(sentences):
    sentence_lengths = [len(s) for s in sentences]
    max_length = max(sentence_lengths)
    batch_size = len(sentences)
    token_arrays = np.zeros([batch_size, max_length], dtype=np.int32)
    for s, s_len, target in zip(sentences, sentence_lengths, token_arrays):
        target[:s_len] = s
    token_lens = np.array(sentence_lengths, dtype=np.int32)
    return token_arrays.astype(np.int32), token_lens


def extract_gt(capt, capt_len):
    gt = []
    for c, c_len in zip(capt, capt_len):
        tmp = [VOCAB_CONFIG.start_token_id] + \
              c[:c_len].tolist() + \
              [VOCAB_CONFIG.end_token_id]
        gt.append(np.array(tmp))
    return gt


def post_proc_yes_no(sampled, gt, ans_seq):
    """
    Replace all yes/no type random sentences with ground truth
    """
    is_yes_no = np.logical_or(ans_seq == 2,
                              ans_seq == 3)
    gt_seqs = extract_gt(*gt)

    final_seq = []
    for t, s, g in zip(is_yes_no, sampled, gt_seqs):
        final_seq.append(g if t else s)
    path, path_len = put_to_array(final_seq)
    return path, path_len, is_yes_no


def train(train_op, model, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, feed_fn=None, env=None):
    feed_train(train_op, model, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               init_fn, saver, reader, feed_fn, env)


def feed_train(train_op, model, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               init_fn, saver, reader=None, feed_fn=None, env=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(graph=graph,
                      config=tf.ConfigProto(gpu_options=gpu_options))
    # initialise training
    ckpt = tf.train.get_checkpoint_state(train_dir)
    sv_path = os.path.join(train_dir, 'model.ckpt')
    if ckpt is None:
        with graph.as_default():
            init_op = tf.initialize_all_variables()
        sess.run(init_op)
        if init_fn is not None:
            init_fn(sess)
    else:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info('Restore from model %s' % os.path.basename(ckpt_path))
        saver.restore(sess, ckpt_path)

    # start reader
    reader.start()
    to_sentence = SentenceGenerator('trainval')

    # customized training code
    for itr in range(number_of_steps):
        if itr % 1000 == 0:
            tf.logging.info('Saving model %s\n' % sv_path)
            saver.save(sess, sv_path, global_step=global_step)
            # pdb.set_trace()

        # get data
        start_time = time.time()

        reader_outputs = reader.pop_batch()
        im, attr, capt, capt_len, ans_seq, ans_seq_len = reader_outputs

        # random sampling
        path = model.random_sampling([im, attr, ans_seq, ans_seq_len], sess)
        rand_pathes = post_process_prediction_default(path)
        rand_path, rand_path_len = put_to_array(rand_pathes)

        # proc yes/no
        # rand_path, rand_path_len, is_yes_no = post_proc_yes_no(rand_pathes,
        #                                                        [capt, capt_len],
        #                                                        ans_seq[:, 0])

        value_inputs = [im, attr, ans_seq, ans_seq_len, rand_path, rand_path_len]
        values = model.run_critic(value_inputs, sess)
        # to_sentence.index_to_question()

        # compute rewards
        rewards = env.get_reward([rand_path, rand_path_len], [capt, capt_len])
        avg_reward = rewards.mean()
        rewards = np.tile(rewards[:, np.newaxis], [1, rand_path.shape[1]-1]).flatten()

        # compute advantage
        advantage = rewards - values

        # proc yes/no
        # batch_size = capt.shape[0]
        # advantage = advantage.reshape([batch_size, -1])
        # advantage[is_yes_no, :] = 1.0
        # advantage = advantage.flatten()

        inputs = [im, attr, rand_path, rand_path_len, ans_seq, ans_seq_len, rewards, advantage]

        total_loss, np_global_step = sess.run([train_op, global_step],
                                              feed_dict=feed_fn(inputs))
        time_elapsed = time.time() - start_time

        if itr % log_every_n_steps == log_every_n_steps - 1:
            tf.logging.info('global step %d: loss = %.3f, average rewards = %0.3f (%.2f sec/step)',
                            np_global_step, total_loss, avg_reward, time_elapsed)

    # Finish training
    tf.logging.info('Finished training! Saving model to disk.')
    saver.save(sess, sv_path, global_step=global_step)

    # Close
    reader.stop()
    sess.close()
