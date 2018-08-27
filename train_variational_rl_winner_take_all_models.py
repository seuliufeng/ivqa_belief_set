from __future__ import division
import tensorflow as tf

from models.model_creater import get_model_creation_fn
from config import TrainConfig, ModelConfig
import var_rl_training_util as training_util
from readers.ivqa_reader_creater import create_reader
from var_ivqa_rewards import IVQARewards, MixReward
from models.language_model import LanguageModel

tf.flags.DEFINE_string("model_type", "VAQ-VarRL",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Dataset version used for training, v1 for VQA 1.0, v2 "
                       "for VQA 2.0.")
tf.flags.DEFINE_string("train_dir", "model/%s_winner_take_all_restval_%s",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 10000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    model_config = ModelConfig()
    training_config = TrainConfig()

    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)
    reader_fn = create_reader('VAQ-Var', phase='train')

    env = MixReward()
    env.diversity_reward.mode = 'winner_take_all'
    env.set_cider_state(False)
    env.set_language_thresh(0.2)

    # Create training directory.
    train_dir = FLAGS.train_dir % (FLAGS.version, FLAGS.model_type)
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'train')
        model.build()

        # Set up the learning rate.u
        learning_rate = tf.constant(training_config.initial_learning_rate * 0.1)

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
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

        # Setup summaries
        summary_op = tf.summary.merge_all()

        # Setup language model
        lm = LanguageModel()
        lm.build()
        env.set_language_model(lm)

    # create reader
    reader = reader_fn(batch_size=16,
                       subset='kprestval',  # 'kptrain'
                       version=FLAGS.version)

    # Run training.
    training_util.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver,
        reader=reader,
        model=model,
        summary_op=summary_op,
        env=env)


def main(_):
    with tf.Graph().as_default():
        train()


if __name__ == '__main__':
    tf.app.run()
