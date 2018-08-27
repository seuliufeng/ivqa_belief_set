from tensorflow.contrib import slim
from util import *
from ops import concat_op
from ivqa_inputs_propressing import process_input_data
from ivqa_sat_decoder import build_decoder
from config import VOCAB_CONFIG


class AttentionModel(object):
    def __init__(self, config, phase='train'):
        self._phase = phase
        self._config = config
        self._keep_prob = self._config.keep_prob if self._phase == 'train' else 1.0
        # self._keep_prob = 1.0
        # override
        self._config.pad_token = VOCAB_CONFIG.pad_token
        self._config.num_dec_cells = 512
        # question
        self._vocab_size = VOCAB_CONFIG.vocab_size
        self._word_embed_dim = self._config.word_embed_dim
        # answer
        self._initializer = get_default_initializer()
        self._answer_embed = None
        self._attribute = None

        self._scores = self._path = None

        # other vars
        self._feed_dict_keys = []
        self._image = self._quest = self._quest_len = self._answers = self._ans_len = None
        self._logits = self._loss = self._im_embed = self.losses = None
        self._global_step = None
        self.init_fn = None
        self.model_vars = []
        self.debug_ops = []

    def build_inputs(self):
        print(self._phase)
        if self._phase == 'train' or self._phase == 'evaluate':
            self._image = tf.placeholder(tf.float32, [None, 14, 14, 2048])
            self._quest = tf.placeholder(tf.int32, [None, None])
            self._quest_len = tf.placeholder(tf.int32, None)
            self._answers = tf.placeholder(tf.int32, [None, None])
            self._ans_len = tf.placeholder(tf.int32, None)
            self._feed_dict_keys = [self._image, self._attribute, self._quest,
                                    self._quest_len, self._answers, self._ans_len]
        elif self._phase == 'condition':
            image_feed = tf.placeholder(tf.float32, [14, 14, 2048])
            attr_feed = tf.placeholder(tf.float32, [1000])
            self._quest = tf.placeholder(tf.int32, [None, None])
            self._quest_len = tf.placeholder(tf.int32, None)
            answers_feed = tf.placeholder(tf.int32, [None])
            ans_len_feed = tf.placeholder(tf.int32, None)
            # replicate inputs
            batch_size = tf.shape(self._quest)[0]
            self._image = tf.tile(tf.expand_dims(image_feed, 0), [batch_size, 1, 1, 1])
            self._attribute = tf.tile(tf.expand_dims(attr_feed, 0), [batch_size, 1])
            self._answers = tf.tile(tf.expand_dims(answers_feed, 0), [batch_size, 1])
            self._ans_len = tf.tile(ans_len_feed, [batch_size])

            self._feed_dict_keys = [image_feed, attr_feed, self._quest, self._quest_len,
                                    answers_feed, ans_len_feed]
        elif self._phase == 'greedy' or self._phase == 'beam':
            self._image = tf.placeholder(tf.float32, [None, 14, 14, 2048])
            self._attribute = tf.placeholder(tf.float32, [None, 1000])
            self._quest = tf.placeholder(tf.int32, [None, None])
            self._quest_len = tf.placeholder(tf.int32, None)
            self._answers = tf.placeholder(tf.int32, [None, None])
            self._ans_len = tf.placeholder(tf.int32, None)
        else:
            raise Exception('unknown mode')

    def build_image_encoder(self):
        self._im_embed = self._image

    def build_answer_basis(self):
        ans_vocab_size = VOCAB_CONFIG.answer_vocab_size
        enc_lstm = create_dropout_lstm_cells(256, self._keep_prob, self._keep_prob)
        # create word embedding
        with tf.variable_scope('answer_embed'):
            ans_embed_map = tf.get_variable(name='word_map', shape=[ans_vocab_size, self._word_embed_dim],
                                            initializer=get_default_initializer())
            ans_word_embed = tf.nn.embedding_lookup(ans_embed_map, self._answers)

        _, states = tf.nn.dynamic_rnn(enc_lstm, ans_word_embed, self._ans_len,
                                      dtype=tf.float32, scope='AnswerEncoder')
        # self.debug_ops.append(ans_word_embed)
        self._answer_embed = concat_op(values=states, axis=1)  # concat tuples and concat

    def build_question_generator(self):
        answer_embed = self._answer_embed
        attr = tf.reduce_mean(tf.reshape(self._image, [-1, 14*14, 2048]), axis=1)
        outputs = build_decoder(self._image, attr,
                                answer_embed,
                                self._quest,
                                self._quest_len,
                                self._vocab_size,
                                self._keep_prob,
                                self._config.pad_token,
                                self._config.num_dec_cells,
                                self._phase)
        if self._phase == 'train' or self._phase == 'condition' or self._phase == 'evaluate':
            self.losses = outputs
        elif self._phase == 'greedy' or self._phase == 'beam':
            self._scores, self._path = outputs

    def collect_model_vars(self):
        self.model_vars = tf.global_variables()

    def setup_global_step(self):
        if self._phase != 'train':
            return
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self._global_step = global_step

    def build_loss(self):
        if self._phase == 'train':
            self._loss = slim.losses.get_total_loss()

    def build(self):
        self.build_inputs()
        self.build_answer_basis()
        self.build_image_encoder()
        self.build_question_generator()
        self.build_loss()
        self.collect_model_vars()
        self.print_variables()
        self.setup_global_step()

    def fill_feed_dict(self, inputs):
        im, attr, capt, capt_len, ans_seq, ans_seq_len = inputs
        proc_inputs = [im, capt, capt_len]
        im, capt, capt_len = process_input_data(proc_inputs, self._config.pad_token)
        inputs = [im, attr, capt, capt_len, ans_seq, ans_seq_len]
        feed_dict = {k: v for (k, v) in zip(self._feed_dict_keys, inputs)}
        return feed_dict

    def greedy_inference(self, inputs, sess):
        im, attr, ans_seq, ans_seq_len = inputs
        return sess.run([self._scores, self._path], feed_dict={self._image: im,
                                                               self._attribute: attr,
                                                               self._answers: ans_seq,
                                                               self._ans_len: ans_seq_len})

    @staticmethod
    def print_variables():
        nparams = 0
        for var in tf.trainable_variables():
            print('%s:' % var.name)
            print(var.get_shape())
            nparams += np.prod(np.array(var.get_shape().as_list()))
        print('Total parameters: %0.2f M.' % (nparams / float(1e6)))
        print('%d vars' % len(tf.trainable_variables()))

    @property
    def loss(self):
        return self._loss

    @property
    def prob(self):
        return self._logits

    @property
    def global_step(self):
        return self._global_step


if __name__ == '__main__':
    from config import ModelConfig
    model = AttentionModel(ModelConfig(), phase='train')
    model.build()
