import tensorflow as tf

from data_fetcher import StateDataPetcher
from util import get_default_initializer
from sequence_mlp import _naive_ranking_loss


tf.flags.DEFINE_string("train_dir", "model/vaq_state_rescore",
                       "Directory for saving and loading model checkpoints.")
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


class StateClassifier(object):
    def __init__(self, input_dim, phase='train'):
        self._phase = phase
        self._input_dim = input_dim
        self._keep_prob = 0.5 if self._phase == 'train' else 1.0
        self._weight_decay = 1e-3
        self._initializer = get_default_initializer()

        # list variables
        self._states = None
        self._labels = None
        self._tot_loss = self._logits = None
        self.global_step = None
        self._feed_keys = None

    def add_logit_regulariser(self):
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('logits/weights')
        return tf.mul(self._weight_decay, tf.nn.l2_loss(weights))

    def build_inputs(self):
        self._states = tf.placeholder(dtype=tf.float32, shape=[None, self._input_dim])
        self._labels = tf.placeholder(dtype=tf.float32, shape=[None])
        self._feed_keys = [self._states]
        if self._phase == 'train':
            self._feed_keys.append(self._labels)

    def fill_feed_dict(self, inputs):
        return {k: v for (k, v) in zip(self._feed_keys, inputs)}

    def build_model(self):
        # compute outputs
        self._logits = tf.contrib.layers.fully_connected(inputs=self._states,
                                                         num_outputs=1,
                                                         activation_fn=None,
                                                         weights_initializer=self._initializer,
                                                         scope='logits')
        self._logits = tf.squeeze(self._logits)
        if self._phase == 'train':
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(self._logits, self._labels)
            losses = _naive_ranking_loss(self._logits, self._labels)
            self._tot_loss = tf.reduce_mean(losses) + self.add_logit_regulariser()
        else:
            self._logits = tf.nn.sigmoid(self._logits)

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self.global_step = global_step

    def build(self):
        self.build_inputs()
        self.build_model()

    @property
    def tot_loss(self):
        return self._tot_loss

    @property
    def prob(self):
        return self._logits


def train():
    import os

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Create model
    max_iter = 1000000
    model = StateClassifier(input_dim=512, phase='train')
    model.build()

    loss = model.tot_loss
    # global_step = model.global_step
    train_op = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    reader = StateDataPetcher(batch_size=18, subset='dev', shuffle=False)
    reader.set_mode('train')
    saver = tf.train.Saver(max_to_keep=5)
    for i in range(max_iter):
        batch_data = reader.pop_batch()
        feed_dict = model.fill_feed_dict(batch_data[:-1])
        _, obj = sess.run([train_op, model.tot_loss], feed_dict=feed_dict)

        if i % 100 == 0:
            tf.logging.info('Iteration %d, loss=%0.2f' % (i, obj))

        if i % 5000 == 0:
            saver.save(sess, os.path.join(FLAGS.train_dir, 'model-%d.ckpt' % i))


def test():
    import json
    import numpy as np
    from w2v_answer_encoder import MultiChoiceQuestionManger

    model = StateClassifier(input_dim=512, phase='test')
    model.build()
    prob = model.prob

    # Load vocabulary
    # to_sentence = SentenceGenerator(trainset='trainval')
    # create multiple choice question manger
    mc_manager = MultiChoiceQuestionManger(subset='val',
                                           answer_coding='sequence')

    sess = tf.Session()
    # Load model
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    checkpoint_path = ckpt.model_checkpoint_path
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # get data
    result = []
    reader = StateDataPetcher(batch_size=18, subset='dev',
                              shuffle=False, max_epoch=1)
    num = reader.num_samples
    for itr in range(num):
        feat, label, quest_id = reader.pop_batch()
        feed_dict = model.fill_feed_dict([feat])
        scores = sess.run(prob, feed_dict=feed_dict)
        idx = scores.argmax()
        # parse question and answer
        assert(np.unique(quest_id).size == 1)
        quest_id = quest_id[0]
        question = mc_manager.get_question(quest_id)
        mc_ans = mc_manager.get_candidate_answers(quest_id)
        vaq_answer = mc_ans[idx]
        real_answer = mc_ans[label.argmax()]
        # add result
        result.append({u'answer': vaq_answer, u'question_id': quest_id})
        # show results
        if itr % 100 == 0:
            print('============== %d ============' % itr)
            print('question id: %d' % quest_id)
            print('question\t: %s' % question)
            print('answer\t: %s' % real_answer)
            print('VAQ answer\t: %s (%0.2f)' % (vaq_answer, scores[idx]))

    quest_ids = [res[u'question_id'] for res in result]
    # save results
    tf.logging.info('Saving results')
    res_file = 'result/rescore_state_dev_dev.json'
    json.dump(result, open(res_file, 'w'))
    from vqa_eval import evaluate_model
    acc = evaluate_model(res_file, quest_ids)
    print ('Over all accuarcy: %0.2f' % acc)
    return acc


if __name__ == '__main__':
    train()
    from time import sleep
    accs = []
    fs = open('result/aux_final_state_rank_res.txt', 'a')
    while True:
        with tf.Graph().as_default():
            acc = test()
        accs.append(acc)
        fs.write('%0.2f\n' % accs[-1])
        sleep(2)
        print(accs)
    fs.close()

