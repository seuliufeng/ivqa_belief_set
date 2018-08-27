from __future__ import division
import tensorflow as tf

import os
import numpy as np
from util import save_hdf5, update_progress
from vqa_model_creater import get_model_creation_fn
from config import ModelConfig
from mtl_data_fetcher import AttentionTestDataFetcher as Reader

tf.flags.DEFINE_string("model_type", "VAQ-van",
                       "Select a model to train.")
tf.flags.DEFINE_string("checkpoint_dir", "model/vaq_%s",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def vaq_condition(checkpoint_path=None):
    subset = 'dev'
    model_config = ModelConfig()

    # Get model
    model_fn = get_model_creation_fn(FLAGS.model_type)

    # build data reader
    reader = Reader(batch_size=1, subset=subset, output_attr=True, output_im=False,
                    output_qa=True, output_capt=False)

    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % FLAGS.model_type)
        checkpoint_path = ckpt.model_checkpoint_path

    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = model_fn(model_config, 'condition')
        model.build()
        saver = tf.train.Saver()

        sess = tf.Session()
        tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
        saver.restore(sess, checkpoint_path)

    fetch_op = model.losses
    num_batches = reader.num_batches

    save_file = 'data/%s_vaq_cond_score1000-2000_%s.hdf5' % ((FLAGS.model_type).lower(), subset)
    print('Save File: %s' % save_file)
    print('Running conditioning...')
    nlls, quest_ids = [], []
    for i in range(num_batches):
        update_progress(i / float(num_batches))

        outputs = reader.get_test_batch()
        im_feed, quest, _, ans_feed, quest_id, image_id = outputs

        losses = sess.run(fetch_op, feed_dict=model.fill_feed_dict(outputs[:-2]))
        scores = losses[:, :-1].mean(axis=1)
        scores = scores[np.newaxis, ::]
        nlls.append(scores)
        quest_ids.append(quest_id)

    nlls = np.concatenate(nlls, axis=0)
    quest_ids = np.concatenate(quest_ids, axis=0)
    print('\nSaving result files: %s...' % save_file)
    save_hdf5(save_file, {'nll': nlls, 'quest_ids': quest_ids})



def main(_):
    vaq_condition()

if __name__ == '__main__':
    tf.app.run()
