from __future__ import division
import tensorflow as tf
import os

# from models.model_creater import get_model_creation_fn
from config import TrainConfig, ModelConfig
# import train_framework_curriculum
import training_util
from readers.vqa_irrelevance_data_fetcher_self import AttentionDataReader as Reader
from models.vqa_soft_attention2_qrd_bruteforce import AttentionModel as model_fn

tf.flags.DEFINE_string("model_type", "VQA-QRD-BF-SLF",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("train_dir", "model/%s_vqa_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    model_config = ModelConfig()
    training_config = TrainConfig()

    # Get model
    # model_fn = get_model_creation_fn(FLAGS.model_type)

    # Create training directory.
    train_dir = FLAGS.train_dir % (FLAGS.version, FLAGS.model_type)
    do_counter_sampling = FLAGS.version == 'v2'
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config,
                         phase='train')
        model.build()

        # Set up the learning rate
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
            learning_rate_decay_fn=learning_rate_decay_fn,
            variables=model.qrd_vars)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

        # setup summaries
        summary_op = tf.summary.merge_all()

    # create reader
    model_name = os.path.split(train_dir)[1]
    reader = Reader(batch_size=model_config.batch_size,
                    subset='trainval',
                    model_name=model_name,
                    epsilon=0.5,
                    feat_type='res5c',
                    version=FLAGS.version,
                    counter_sampling=do_counter_sampling)

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
        summary_op=summary_op)


def main(_):
    with tf.Graph().as_default():
        train()


if __name__ == '__main__':
    tf.app.run()
