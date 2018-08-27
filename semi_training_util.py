import tensorflow as tf
import os
import time
import numpy as np


def train(train_op, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, model=None,
          summary_op=None, sync_op=None):
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
                   sync_op)


def feed_train(train_op, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               init_fn, saver, reader=None, model=None,
               summary_op=None, sync_op=None):
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

    # initialise training
    ckpt = tf.train.get_checkpoint_state(train_dir)
    sv_path = os.path.join(train_dir, 'model.ckpt')
    with graph.as_default():
        init_op = tf.initialize_all_variables()
    sess.run(init_op)
    if ckpt is None:
        if init_fn is not None:
            init_fn(sess)
    else:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info('Restore from model %s' % os.path.basename(ckpt_path))
        saver.restore(sess, ckpt_path)

    # start reader
    reader.start()

    feed_fn = model.fill_training_feed_dict
    # customized training code
    for itr in range(number_of_steps):
        if itr % 1000 == 0:
            tf.logging.info('Saving model %s\n' % sv_path)
            saver.save(sess, sv_path, global_step=global_step)
        if itr % 2000 == 0:
            if sync_op is not None:
                tf.logging.info('Synchronising models\n')
                sess.run(sync_op)
        start_time = time.time()
        batch_data = reader.pop_batch()
        # sampling
        smp_labels, rewards, noise = model.sampling(sess, batch_data)
        # process data
        top_labels = batch_data[-1]
        unlabel_tab = top_labels == -1
        label_tab = top_labels != -1
        top_labels[unlabel_tab] = smp_labels[unlabel_tab]
        pos_mean = rewards[label_tab].mean()
        rewards[label_tab] = 1.0
        rewards[rewards < pos_mean] = 0.
        rewards[np.logical_and(rewards >= pos_mean, unlabel_tab)] = 0.9
        batch_data += [noise, rewards]
        # pdb.set_trace()
        if _write_summary and itr % summary_interval == 0:
            total_loss, np_global_step, summary = sess.run([train_op, global_step, summary_op],
                                                           feed_dict=feed_fn(batch_data))
            summary_writer.add_summary(summary, np_global_step)
        else:
            total_loss, np_global_step = sess.run([train_op, global_step],
                                                  feed_dict=feed_fn(batch_data))

        time_elapsed = time.time() - start_time

        if itr % log_every_n_steps == log_every_n_steps - 1:
            tf.logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                            np_global_step, total_loss, time_elapsed)

    # Finish training
    tf.logging.info('Finished training! Saving model to disk.')
    saver.save(sess, sv_path, global_step=global_step)

    # Close
    reader.stop()
    sess.close()
