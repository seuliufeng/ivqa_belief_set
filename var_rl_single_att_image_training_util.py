import tensorflow as tf
import os
import time
# from rl_trainstep import reinforce_trainstep
from rl_single_att_trainstep import reinforce_trainstep
from rl_single_att_trainstep import VQABelief
from post_process_variation_questions import _parse_gt_questions
from write_examples import ExperimentWriter

IM_ROOT = '/usr/data/fl302/data/VQA/Images/'

_WRITER = ExperimentWriter('latex/examples_att_online')


def train(train_op, train_dir, log_every_n_steps,
          graph, global_step, number_of_steps,
          init_fn, saver, reader=None, model=None,
          summary_op=None, env=None):
    try:
        feed_train(train_op, train_dir, log_every_n_steps,
                   graph, global_step, number_of_steps,
                   init_fn, saver, reader, model, summary_op,
                   env)
    except KeyboardInterrupt:
        print('Compiling')
        _WRITER.render()


def feed_train(train_op, train_dir, log_every_n_steps,
               graph, global_step, number_of_steps,
               init_fn, saver, reader=None, model=None,
               summary_op=None, env=None):
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
    lm = env.lm
    lm.set_session(sess)
    # initialise training
    ckpt = tf.train.get_checkpoint_state(train_dir)
    sv_path = os.path.join(train_dir, 'model.ckpt')
    with graph.as_default():
        init_op = tf.initialize_all_variables()
    sess.run(init_op)
    if ckpt is None:
        if init_fn is not None:
            init_fn(sess)
        lm.setup_model()
    else:
        ckpt_path = ckpt.model_checkpoint_path
        tf.logging.info('Restore from model %s' % os.path.basename(ckpt_path))
        saver.restore(sess, ckpt_path)
        lm.setup_model()

    # build belief buffer
    _VQA_Belief = VQABelief()
    # customized training code
    for itr in range(number_of_steps):
        datum = reader.get_test_batch()
        quest_id = datum[-2][0]
        image_id = datum[-1][0]
        top_ans_id = datum[4][0]
        if top_ans_id == 2000:
            continue

        _, _, quest, quest_len, _, ans, ans_len, _, _ = datum
        question = env.to_sentence.index_to_question(_parse_gt_questions(quest, quest_len)[0])
        answer = env.to_sentence.index_to_answer(_parse_gt_questions(ans, ans_len)[0])
        im_file = '%s2014/COCO_%s2014_%012d.jpg' % ('val', 'val', image_id)
        im_path = os.path.join(IM_ROOT, im_file)

        print('Hacking question %d (%d/%d)...' % (quest_id, itr, number_of_steps))
        head = 'Q: %s A: %s' % (question, answer)
        print(head)
        t = time.time()
        while True:
            task_ops = [train_op, global_step]
            total_loss, np_global_step, avg_reward, t_str = \
                reinforce_trainstep(datum, model, env, sess, task_ops, _VQA_Belief)
            if _VQA_Belief.should_terminate():
                break
        print('Hacking finished in %0.2fs' % (time.time()-t))
        questions = _VQA_Belief.show_belief(env, quest_id)

        _WRITER.add_result(image_id, quest_id, im_path, head, questions)
        _VQA_Belief.clear()
        # reset model
        init_fn(sess)

    # Finish training
    # tf.logging.info('Finished training! Saving model to disk.')
    # saver.save(sess, sv_path, global_step=global_step)

    # Close
    # reader.stop()
    sess.close()
