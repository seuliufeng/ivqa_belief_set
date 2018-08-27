import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from readers.ivqa_reader_creater import create_reader
from rerank_feature_fetcher import RerankContext
from rerank_feature_fetcher_v1 import RerankContext as RerankContext_v1
import pdb


def post_process_batch(inputs):
    attr, q, q_len, cands, vqa_scores, ivqa_scores, label, quest_ids = inputs
    is_valid = np.any(np.equal(cands, label[:, np.newaxis]), axis=1)
    outputs = [e[is_valid] for e in inputs]
    return outputs


class Reader(object):
    def __init__(self, batch_size, subset='kpval', phase='train', version='v2'):
        # reader_fn = create_reader('Fusion%s' % ('v1' if version=='v1' else ''), phase=phase)
        reader_fn = create_reader('Fusion%s' % ('v1' if version == 'v1' else ''), phase=phase)
        self.phase = phase
        self.version = version
        self.feat_reader = reader_fn(batch_size, subset, version=version)
        if version == 'v2':
            self.score_reader = RerankContext(subset=subset)
        else:
            self.score_reader = RerankContext_v1(subset=subset)

    def pop_batch(self):
        if self.phase == 'train':
            attr, q, q_len, label, quest_ids = self.feat_reader.pop_batch()
        else:
            attr, q, q_len, label, quest_ids, _ = self.feat_reader.get_test_batch()
        # get scores
        cands, vqa_scores, ivqa_scores = self.score_reader.get_scores(quest_ids)
        outputs = [attr, q, q_len, cands, vqa_scores, ivqa_scores, label, quest_ids]
        if self.phase == 'train':
            outputs = post_process_batch(outputs)
        return outputs

    def start(self):
        if self.phase == 'train':
            self.feat_reader.start()

    def stop(self):
        if self.phase == 'train':
            self.feat_reader.stop()

    @property
    def num_batches(self):
        return self.feat_reader.num_batches


# ------------- model util functions ------------------
def softmax_norm(scores, temp):
    return tf.nn.softmax(scores * temp)


def encode_states(embedding, scores, keep_prob, scope):
    with tf.variable_scope(scope):
        scores = tf.expand_dims(scores, 2)
        # scores = slim.fully_connected(scores, 300, activation_fn=tf.nn.tanh, scope='ups')
        avg_embed = tf.reduce_sum(embedding * scores, axis=1)
        state = slim.fully_connected(avg_embed, 512, activation_fn=tf.nn.tanh,
                                     scope='hidden')
        return tf.nn.dropout(state, keep_prob)


def encode_states_rnn(embedding, scores, keep_prob, scope):
    from util import create_dropout_basic_lstm_cells
    with tf.variable_scope(scope):
        scores = tf.expand_dims(scores, 2)
        embedding = embedding * scores
        lstm = create_dropout_basic_lstm_cells(512, keep_prob, keep_prob)
        _, states = tf.nn.dynamic_rnn(lstm, embedding,
                                      dtype=tf.float32, scope='score_rnn')
        return states[1]


def attention_network(inputs, num_hid=512, keep_prob=0.5, reuse=False, scope='attention'):
    with tf.variable_scope(scope, reuse=reuse):
        in_cat = tf.concat(inputs, axis=1)
        in_red = slim.fully_connected(in_cat, num_hid,
                                      activation_fn=tf.nn.tanh,
                                      scope='hidden')
        in_red = slim.dropout(in_red, keep_prob=keep_prob)
        return slim.fully_connected(in_red, 1, activation_fn=None,
                                    scope='logits')


def xe_loss(preds, targets, mask, EPS=1e-8):
    neg_xe = tf.log(preds + EPS) * targets + \
             tf.log(1.0 - preds + EPS) * (1.0 - targets)
    neg_xe = tf.div(tf.reduce_sum(neg_xe * mask), tf.reduce_sum(mask))
    return -neg_xe


def ranking_loss(preds, gt_mask, margin=0.1):
    batch_size = tf.shape(preds)[0]
    pos = tf.boolean_mask(preds, gt_mask)
    pos = tf.reshape(pos, [batch_size, 1])
    cst_mask = 1.0 - tf.cast(gt_mask, tf.float32)  # negative samples
    losses = tf.nn.relu(preds + margin - pos) * cst_mask
    return tf.div(tf.reduce_sum(losses), tf.reduce_sum(cst_mask))


def entropy_loss(gates, EPS=1e-8):
    neg_ent = tf.reduce_mean(tf.reduce_sum(tf.log(gates + EPS) * gates,
                                           axis=1))
    return 0.7 + neg_ent


def add_gaussian_noise(inputs, std):
    z = tf.random_normal(inputs.get_shape())
    return inputs + z * std


# ------------ rerank model ---------------------
class RerankModel(object):
    def __init__(self, phase='train', version='v2', num_cands=3):
        self.keep_prob = 0.5 if phase == 'train' else 1.0
        self.gate_keep_prob = 0.8 if phase == 'train' else 1.0
        # self._im_dim = 2048 if version == 'v2' else 1000
        self._im_dim = 2048
        self.num_cands = num_cands

    def _build_inputs(self):
        # context
        self._attribute = tf.placeholder(tf.float32, [None, self._im_dim])
        self._quest = tf.placeholder(tf.int32, [None, None])
        self._quest_len = tf.placeholder(tf.int32, None)

        # states
        K = self.num_cands
        self._cands = tf.placeholder(tf.int32, [None, K])
        self._vqa_scores = tf.placeholder(tf.float32, [None, K])
        self._ivqa_scores = tf.placeholder(tf.float32, [None, K])

        # ground truth
        self._label = tf.placeholder(tf.int32, None)
        self._feed_dict_keys = [self._attribute, self._quest, self._quest_len,
                                self._cands, self._vqa_scores, self._ivqa_scores,
                                self._label]

    def _build_question_encoder(self):
        from skip_thought_util.skip_thought_model import SkipThoughtEncoder
        seq_encoder = SkipThoughtEncoder(15954, self.keep_prob)
        enc_output = seq_encoder(self._quest, self._quest_len)
        self.quest_embed = slim.fully_connected(enc_output, 512, activation_fn=tf.nn.tanh,
                                                scope='quest_embed')
        self.quest_embed = slim.dropout(self.quest_embed, self.keep_prob)
        self.init_fn = seq_encoder.setup_initializer('skip_thought_util/skip_thought_16k.npy')

    def _build_image_encoder(self):
        self.im_embed = slim.fully_connected(self._attribute, 512,
                                             activation_fn=tf.nn.tanh,
                                             scope='image_embedding')
        self.im_embed = slim.dropout(self.im_embed, self.keep_prob)

    def _build_label_embedding(self):
        from util import load_hdf5
        d = load_hdf5('../iccv_vaq/data/top_answers_2000_w2v.h5')
        word2vec = d['word2vec']  # 2000x300
        word2vec = np.concatenate([word2vec, np.zeros((1, 300), dtype=np.float32)], axis=0)

        with tf.variable_scope('label_embedding'):
            label_map = tf.Variable(initial_value=word2vec, trainable=True,
                                    dtype=tf.float32, name='map')
            # label_map = tf.get_variable('map', shape=[2001, 300],
            #                             dtype=tf.float32,
            #                             initializer=get_default_initializer())
        self.cand_embed = tf.nn.embedding_lookup(label_map, self._cands)

    def _build_gates(self):
        # context states
        ctx = tf.concat([self.im_embed, self.quest_embed], axis=1)
        ctx = slim.fully_connected(ctx, 512, activation_fn=tf.nn.tanh, scope='context')
        ctx = tf.nn.dropout(ctx, self.keep_prob)
        # compute temp
        temp = slim.fully_connected(ctx, 1, activation_fn=tf.nn.softplus, scope='softmax_temp')
        self._norm_ivqa_scores = softmax_norm(self._ivqa_scores, temp)
        # submodel states
        state_fn = encode_states
        # state_fn = encode_states_rnn
        vqa_states = state_fn(self.cand_embed, self._vqa_scores,
                              self.keep_prob, 'state_vqa')
        ivqa_state_in = self._norm_ivqa_scores  # [default]
        # ivqa_state_in = self._ivqa_scores
        ivqa_states = state_fn(self.cand_embed, ivqa_state_in,
                               self.keep_prob, 'state_ivqa')

        # state embedding
        # independent
        comp_vqa = attention_network([ctx, vqa_states], keep_prob=self.keep_prob, scope='comp_vqa')
        comp_ivqa = attention_network([ctx, ivqa_states], keep_prob=self.keep_prob, scope='comp_ivqa')
        # shared
        # comp_vqa = attention_network([ctx, vqa_states], reuse=False, scope='attention')
        # comp_ivqa = attention_network([ctx, ivqa_states], reuse=True, scope='attention')
        gates = tf.concat([comp_vqa, comp_ivqa], axis=1)
        self.gates = tf.nn.softmax(gates)
        # self.gates = tf.nn.dropout(gates, self.gate_keep_prob)

    def _fuse_outputs(self):
        gate_vqa, gate_ivqa = tf.split(self.gates, 2, axis=1)
        # gate_vqa, gate_ivqa = 0.5, 0.5
        # self.preds = self._vqa_scores * gate_vqa + self._norm_ivqa_scores * gate_ivqa
        self.preds = self._vqa_scores + self._norm_ivqa_scores

    def _build_loss(self):
        batch_size = tf.shape(self._quest)[0]
        # targets
        gt = tf.expand_dims(self._label, axis=1)
        gt_mask = tf.reshape(tf.equal(self._cands, gt), [batch_size, self.num_cands])
        targets = tf.cast(gt_mask, tf.float32)
        # mask
        # mask = tf.greater(tf.reduce_sum(targets, axis=1), 0)
        # mask = tf.tile(tf.expand_dims(mask, axis=1), [1, 3])
        # mask = tf.reshape(tf.cast(mask, tf.float32), [batch_size, 3])
        # ent = entropy_loss(self.gates)
        # main_loss = xe_loss(self.preds, targets, mask)
        # # self.loss = main_loss
        # aux_loss = xe_loss(self._norm_ivqa_scores, targets, mask)
        # # self.loss = 0.5 * (main_loss + aux_loss) + 0.0 * ent

        rank_main = ranking_loss(self.preds, gt_mask)
        rank_aux = ranking_loss(self._norm_ivqa_scores, gt_mask)
        self.loss = rank_main + rank_aux

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self.global_step = global_step

    def build(self):
        self._build_inputs()
        self._build_image_encoder()
        self._build_question_encoder()
        self._build_label_embedding()
        self._build_gates()
        self._fuse_outputs()
        self._build_loss()
        self.setup_global_step()

    def fill_feed_dict(self, inputs):
        feed_dict = {k: v for (k, v) in zip(self._feed_dict_keys, inputs[:-1])}
        return feed_dict


def test_reader():
    reader = Reader(4, 'kpval')
    reader.start()
    for i in range(10):
        outputs = reader.pop_batch()
    reader.stop()


def test_model():
    model = RerankModel(phase='train')
    model.build()


if __name__ == '__main__':
    test_model()
    # test_reader()
