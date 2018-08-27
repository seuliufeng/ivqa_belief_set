import tensorflow as tf
import os
import time


def train(train_op, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, feed_fn=None,
          loss_op=None,
          summary_op=None,
          summary_interval=100):
    sess = tf.Session(graph=graph)

    # prepare summary writer
    _write_summary = summary_op is not None
    if _write_summary:
        summary_dir = os.path.join(train_dir, 'summary')
        if not tf.gfile.IsDirectory(summary_dir):
            tf.logging.info("Creating summary directory: %s", summary_dir)
            tf.gfile.MakeDirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir)
        # summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

    # initialise training
    ckpt = tf.train.get_checkpoint_state(train_dir)
    sv_path = os.path.join(train_dir, 'model.ckpt')
    # first try to initialise all variables
    with graph.as_default():
        init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # if there're specific init functions, ...
    if ckpt is None:
        if init_fn is not None:
            init_fn(sess)
    else:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info('Restore from model %s' % os.path.basename(ckpt_path))
        saver.restore(sess, ckpt_path)

    # start reader
    reader.start()

    # customized training code
    for itr in range(number_of_steps):
        if itr % 5000 == 0:
            tf.logging.info('Saving model %s\n' % sv_path)
            saver.save(sess, sv_path, global_step=global_step)
            reader.backup_statistics()
        start_time = time.time()
        if _write_summary and itr % summary_interval == 0:
            total_loss, np_global_step, smp_losses, summary = sess.run([train_op, global_step,
                                                                        loss_op, summary_op],
                                                                       feed_dict=feed_fn(reader.pop_batch()))
            summary_writer.add_summary(summary, np_global_step)
        else:
            total_loss, np_global_step, smp_losses = sess.run([train_op, global_step, loss_op],
                                                              feed_dict=feed_fn(reader.pop_batch()))
        reader.update_loss(smp_losses)  # update loss info for reader
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
