import tensorflow as tf
import tensorflow.contrib.slim as slim
from naive_vqa_agent import NavieVQAAgent


class NaiveEnsembleModel(object):
    def __init__(self, config, phase='train'):
        self._num_agents = 8
        self._phase = phase
        self._agent_embed_size = 256
        self._vocab_size = config.vocab_size
        self._keep_prob = 0.5 if self._phase == 'train' else 1.0

        # inputs
        self._image = tf.placeholder(tf.float32, [None, 2048])
        self._quest = tf.placeholder(tf.int32, [None, None])
        self._quest_len = tf.placeholder(tf.int32, None)
        self._labels = tf.placeholder(tf.int32, None)
        self._feed_dict_keys = [self._image, self._quest, self._quest_len, self._labels]

        #
        self.prob = None
        self.loss = None
        self.global_step = None
        self.init_fn = None
        self.setup_global_step()

    def build(self):
        agents = []
        probs = []
        for i in range(self._num_agents):
            agent_name = 'agent%02d' % i
            agent = NavieVQAAgent(self._feed_dict_keys, self._vocab_size,
                                  self._agent_embed_size, keep_prob=self._keep_prob,
                                  scope=agent_name)
            agent.build_model()
            probs.append(agent.prob)
            agents.append(agent)
        # get loss
        self.loss = slim.losses.get_total_loss()
        self.prob = tf.add_n(probs)
        # print
        self.print_variables()
        self.print_losses()

    def fill_feed_dict(self, inputs):
        feed_dict = {k: v for (k, v) in zip(self._feed_dict_keys, inputs)}
        return feed_dict

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self.global_step = global_step

    @staticmethod
    def print_variables():
        print('Trainable variables:')
        for var in tf.trainable_variables():
            print('%s:' % var.name)
            print(var.get_shape())

    @staticmethod
    def print_losses():
        print('Losses:')
        for loss in slim.losses.get_losses():
            print('%s:' % loss.name)