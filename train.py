from __future__ import division
import tensorflow as tf
from vqa_model_creater import get_model_creation_fn
from config import TrainConfig, ModelConfig

tf.flags.DEFINE_string("model_type", "VQA-Mem",
                       "Select a model to train.")
tf.flags.DEFINE_string("input_file_pattern", "/data/fl302/data/VQA/dbs/res5c/vqa_Res152_mscoco_trainval_zlib.tfrecords",
                       "TFRecord file of training data.")
tf.flags.DEFINE_string("train_dir", "model/res152_%s_res5c",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 250000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    model_config = ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    training_config = TrainConfig()

    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # Create training directory.
    train_dir = FLAGS.train_dir % FLAGS.model_type
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'train')
        model.build()

        # Set up the learning rate.
        learning_rate = tf.constant(training_config.initial_learning_rate)

        def _learning_rate_decay_fn(learn_rate, global_step):
            return tf.train.exponential_decay(
                learn_rate,
                global_step,
                decay_steps=3,
                decay_rate=training_config.decay_factor)

        learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up gradient clipping function
        # def _clip_gradient_by_value(gvs):
        #     return [(tf.clip_by_value(grad, -training_config.clip_gradients,
        #                               training_config.clip_gradients), var) for grad, var in gvs]
        # grad_proc_fn = _clip_gradient_by_value

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver)


def main(_):
    with tf.Graph().as_default():
        train()


if __name__ == '__main__':
    tf.app.run()
