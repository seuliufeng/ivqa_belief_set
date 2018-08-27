import tensorflow as tf
import tensorflow.contrib.slim as slim
from deep_lstm_encoder import DeepLSTMEncoder


class NavieVQAAgent(object):
    def __init__(self, inputs, vocab_size, embed_size=256, num_outputs=2001,
                 keep_prob=1.0, scope='AgentVQA'):
        self._embed_size = embed_size
        self._num_classes = num_outputs
        self._keep_prob = keep_prob
        self._build_loss = len(inputs) == 4
        self._vocab_size = vocab_size
        if self._build_loss:
            self._image, self._quest, self._quest_len, self._labels = inputs
        else:
            self._image, self._quest, self._quest_len = inputs
        self._scope = scope
        self._quest_enc = DeepLSTMEncoder(self._vocab_size)
        self._logits = None
        self.prob = None

    def build_model(self):
        with tf.variable_scope(self._scope):
            # question
            quest = self._quest_enc(self._quest, self._quest_len)  # 1024d
            quest = slim.fully_connected(quest, self._embed_size, tf.nn.tanh,
                                         scope='Q')
            # image
            image = tf.nn.l2_normalize(self._image, dim=1)
            image = slim.fully_connected(image, self._embed_size, tf.nn.tanh,
                                         scope='I')
            # joint embedding
            joint = slim.dropout(quest * image, self._keep_prob)
            pre_logits = slim.fully_connected(joint, self._embed_size, tf.nn.tanh,
                                              scope='MLP1')
            pre_logits = slim.dropout(pre_logits, self._keep_prob)
            # logits
            logits = slim.fully_connected(pre_logits, self._num_classes, None,
                                          scope='logits')
            self._logits = logits
            self.prob = tf.nn.softmax(self._logits)
            if self._build_loss:
                self.build_loss()

    def build_loss(self):
        slim.losses.sparse_softmax_cross_entropy(self._logits, self._labels,
                                                 scope=self._scope+'_loss')
