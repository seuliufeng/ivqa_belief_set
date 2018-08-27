# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from config import QuestionGeneratorConfig
from vqa_question_generator import QuestionGenerator
from record_reader import TFRecordDataFetcher
from inference_utils.question_generator_util import SentenceGenerator

tf.flags.DEFINE_string("checkpoint_dir", "model/vaq_nic_config_trainval_incept",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
# tf.flags.DEFINE_string("checkpoint_dir", "model/question_gen_trainval",
#                        "Model checkpoint file or directory containing a "
#                        "model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "trainval",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("input_files", "data/vqa_incept_mscoco_dev.tfrecords",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
# tf.flags.DEFINE_string("input_files", "data/vqa_mscoco_dev.tfrecords",
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def restore_model(sess, checkpoint_path):
    saver = tf.train.Saver(var_list=tf.all_variables())
    saver.restore(sess, checkpoint_path)


def post_processing_data(reader_outs):
    im_ids, quest_id, im_feat, ans_w2v, quest_ids, ans_ids = reader_outs
    im_feat = im_feat.reshape([1, -1])
    ans_w2v = ans_w2v.reshape([1, -1])
    quest_in = np.array(quest_ids[:-1]).reshape([1, -1])
    quest_targ = np.array(quest_ids[1:]).reshape([1, -1])
    quest_mask = np.array(np.ones_like(quest_targ), dtype=np.int64)
    return im_feat, ans_w2v, quest_in, quest_targ, quest_mask


def main(_):
    # Build the inference graph.
    config = QuestionGeneratorConfig()
    reader = TFRecordDataFetcher(FLAGS.input_files,
                                 config.image_feature_key)

    g = tf.Graph()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)
    with g.as_default():
        model = QuestionGenerator(config, phase='evaluate')
        model.build()
    # g.finalize()

    # Create the vocabulary.
    to_sentence = SentenceGenerator(trainset=FLAGS.model_trainset)

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        saver = tf.train.Saver(var_list=tf.all_variables())
        saver.restore(sess, checkpoint_path)

        itr = 0
        while not reader.eof():
            outputs = reader.pop_batch()
            im_ids, quest_id, im_feat, ans_w2v, quest_ids, ans_ids = outputs
            inputs = post_processing_data(outputs)
            perplexity = sess.run(model.likelihood,
                                  feed_dict=model.fill_feed_dict(inputs))

            # generated = [generated[0]]  # sample 3
            question = to_sentence.index_to_question(quest_ids)
            answer = to_sentence.index_to_answer(ans_ids)

            print('============== %d ============' % itr)
            print('image id: %d, question id: %d' % (im_ids, quest_id))
            print('question\t: %s' % question)
            elems = question.split(' ')
            tmp = ' '.join(['%s (%0.2f)' % (w, p) for w, p in zip(elems, perplexity.flatten())][:-1])
            print('question\t' + tmp)
            print('answer\t: %s' % answer)
            print ('perplexity\t: %0.2f\n' % perplexity.mean())

            itr += 1


if __name__ == "__main__":
    tf.app.run()
