from __future__ import division
import tensorflow as tf
# from rerank_fusion import RerankModel, Reader
from rerank_fusion_v1 import RerankModel, Reader
# from rerank_fusion_v2 import RerankModel, Reader
import training_util
from config import TrainConfig


tf.flags.DEFINE_string("train_dir", "/import/vision-ephemeral/fl302/data/model/%s_kpvaq1_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 100000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    training_config = TrainConfig()
    # Create training directory.
    train_dir = FLAGS.train_dir % ('v1', 'Fusion')
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = RerankModel('train', version='v1', num_cands=5)
        model.build()

        # Set up the learning rate.u
        learning_rate = tf.constant(training_config.initial_learning_rate)

        def _learning_rate_decay_fn(learn_rate, global_step):
            return tf.train.exponential_decay(
                learn_rate,
                global_step,
                decay_steps=training_config.decay_step,
                decay_rate=training_config.decay_factor, staircase=False)

        learning_rate_decay_fn = _learning_rate_decay_fn

        train_op = tf.contrib.layers.optimize_loss(
            loss=model.loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        var_list = [var for var in tf.global_variables() if 'Adam' not in var.name]
        saver = tf.train.Saver(var_list, max_to_keep=training_config.max_checkpoints_to_keep)

    # create reader
    # reader = Reader(batch_size=32, subset='trainval', version='v1')
    reader = Reader(batch_size=128*4, subset='trainval', version='v1')

    # Run training.
    training_util.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver, reader=reader,
        feed_fn=model.fill_feed_dict,
        debug_op=None)


def main(_):
    with tf.Graph().as_default():
        train()


if __name__ == '__main__':
    tf.app.run()


