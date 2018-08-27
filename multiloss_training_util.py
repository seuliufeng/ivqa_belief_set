import tensorflow as tf
import os
import time


def train(train_op, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, feed_fn=None,
          loss_ops=None):
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
                   init_fn, saver, reader, feed_fn, loss_ops)


def feed_train(train_op, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               init_fn, saver, reader=None, feed_fn=None,
               loss_ops=None):
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

    # customized training code
    for itr in range(number_of_steps):
        if itr % 3000 == 0:
            tf.logging.info('Saving model %s\n' % sv_path)
            saver.save(sess, sv_path, global_step=global_step)
        start_time = time.time()
        outputs = sess.run([train_op, global_step] + loss_ops,
                           feed_dict=feed_fn(reader.pop_batch()))
        total_loss, np_global_step = outputs[:2]

        time_elapsed = time.time() - start_time

        if itr % log_every_n_steps == log_every_n_steps - 1:
            formated_loss = _print_losses(loss_ops, outputs[2:])
            tf.logging.info('global step %d: %s, tot: %.4f (%.2f sec/step)',
                            np_global_step, formated_loss, total_loss, time_elapsed)

    # Finish training
    tf.logging.info('Finished training! Saving model to disk.')
    saver.save(sess, sv_path, global_step=global_step)

    # Close
    reader.stop()
    sess.close()


def _print_losses(loss_ops, losses):
    print_info = []
    for tf_loss, loss in zip(loss_ops, losses):
        loss_name = tf_loss.name.split(':')[0]
        # remove scope name
        loss_name = loss_name.split('/')[-1]
        loss_str = '%s: %0.4f' % (loss_name, loss)
        print_info.append(loss_str)
    return ', '.join(print_info)
