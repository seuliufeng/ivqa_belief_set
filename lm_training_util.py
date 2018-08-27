import tensorflow as tf
import os
import numpy as np
import time
from post_process_variation_questions import post_process_variation_questions_noise, \
    wrap_samples_for_language_model, _parse_gt_questions
from inference_utils.question_generator_util import SentenceGenerator

_SENT = SentenceGenerator(trainset='trainval')


class QuestionContext(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        from util import load_hdf5
        data_file = 'data/vqa_std_mscoco_kprestval.data'
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        if self._quest.shape[1] > 20:  # truncate by 20
            self._quest = self._quest[:, :20]
            self._quest_len = np.minimum(self._quest_len, 20)
        self.num = self._quest_len.size

    def get_random_questions(self):
        index = np.random.choice(self.num, size=(self.batch_size,),
                                 replace=False)
        return self._quest[index], self._quest_len[index]

    def get_gt_batch(self, cur_arr, cur_arr_len):
        other_arr, other_arr_len = self.get_random_questions()
        arr = self._concat_arrays(cur_arr, other_arr)
        arr_len = np.concatenate([cur_arr_len, other_arr_len], axis=0)
        return arr, arr_len

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


_Q_CTX = QuestionContext(batch_size=12)


def lm_trainstep(train_step, global_step, reader, model, sampler, sess):
    outputs = reader.pop_batch()
    images, quest, quest_len, ans, ans_len = outputs
    quest, quest_len = _Q_CTX.get_gt_batch(quest, quest_len)  # random sample new

    # random sampling
    noise_vec, pathes, scores = sampler.random_sampling([images, ans, ans_len], sess)
    _this_batch_size = images.shape[0]
    scores, pathes, noise = post_process_variation_questions_noise(scores,
                                                                   pathes,
                                                                   noise_vec,
                                                                   _this_batch_size)
    # update language model
    # fake = wrap_samples_for_language_model(pathes)
    # real = [quest, quest_len]
    # lm_inputs = fake + real
    if 'CNN' in model.name:
        lm_inputs = wrap_samples_for_language_model(sampled=pathes,
                                                    pad_token=sampler.pad_token - 1,
                                                    gts=[quest, quest_len],
                                                    max_length=20)
    else:
        lm_inputs = wrap_samples_for_language_model(sampled=pathes,
                                                    pad_token=sampler.pad_token - 1,
                                                    gts=[quest, quest_len])

    loss, gstep = sess.run([train_step, global_step],
                           feed_dict=model.fill_feed_dict(lm_inputs))

    if gstep % 500 == 0:

        def _show_examples(arr, arr_len, name):
            _rewards = model.inference([arr, arr_len])
            ps = _parse_gt_questions(arr, arr_len)
            print('\n%s:' % (name))
            for p, r in zip(ps, _rewards):
                if p[-1] == 2:
                    p = p[:-1]
                sent = _SENT.index_to_question(p)
                print('%s (%0.3f)' % (sent, r))

        fake, fake_len, real, real_len = lm_inputs
        _show_examples(fake, fake_len, 'Fake')
        _show_examples(real, real_len, 'Real')
    return loss, gstep


def train(train_op, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, model=None,
          summary_op=None, sampler=None):
    if reader is None:
        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            train_dir,
            log_every_n_steps=log_every_n_steps,
            graph=graph,
            global_step=global_step,
            number_of_steps=number_of_steps,
            init_fn=init_fn,
            saver=saver)
    else:
        feed_train(train_op, train_dir, log_every_n_steps,
                   graph, global_step, number_of_steps,
                   init_fn, saver, reader, model, summary_op,
                   sampler)


def feed_train(train_op, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               sampler_init_fn, saver, reader=None, model=None,
               summary_op=None, sampler=None):
    summary_writer = None
    sess = tf.Session(graph=graph)
    summary_interval = 100
    # prepare summary writer
    _write_summary = summary_op is not None
    if _write_summary:
        summary_dir = os.path.join(train_dir, 'summary')
        if not tf.gfile.IsDirectory(summary_dir):
            tf.logging.info("Creating summary directory: %s", summary_dir)
            tf.gfile.MakeDirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir)
    # setup language model
    model.set_session(sess)
    # initialise training
    ckpt = tf.train.get_checkpoint_state(train_dir)
    sv_path = os.path.join(train_dir, 'model.ckpt')
    with graph.as_default():
        init_op = tf.initialize_all_variables()
    sess.run(init_op)
    if ckpt is None:
        if sampler_init_fn is not None:
            sampler_init_fn(sess)
        model.setup_model()
    else:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info('Restore from model %s' % os.path.basename(ckpt_path))
        saver.restore(sess, ckpt_path)
        sampler_init_fn(sess)

    # start reader
    reader.start()

    # customized training code
    for itr in range(number_of_steps):
        if itr % 2000 == 0:
            tf.logging.info('Saving model %s\n' % sv_path)
            saver.save(sess, sv_path, global_step=global_step)
        start_time = time.time()

        total_loss, np_global_step = lm_trainstep(train_op, global_step,
                                                  reader, model, sampler,
                                                  sess)

        time_elapsed = time.time() - start_time

        if itr % log_every_n_steps == log_every_n_steps - 1 or itr == 0:
            tf.logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                            np_global_step, total_loss, time_elapsed)

    # Finish training
    tf.logging.info('Finished training! Saving model to disk.')
    saver.save(sess, sv_path, global_step=global_step)

    # Close
    reader.stop()
    sess.close()
