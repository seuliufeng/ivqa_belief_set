import tensorflow as tf
from util import get_default_initializer

if __name__ == "__main__":
    tf.flags.DEFINE_string("train_dir", "model/vaq_rescore",
                           "Directory for saving and loading model checkpoints.")
    FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


class MLPConfig(object):
    def __init__(self):
        self.vocab_size = 3001
        self.word_embed_dim = 300


def _naive_ranking_loss(logits, labels):
    pos = tf.reduce_sum(tf.mul(logits, labels))
    d = tf.mul(1-labels, tf.sub(pos, logits))  # only compute wrt negative
    margin = tf.sub(tf.convert_to_tensor(1.0, dtype=tf.float32), d)
    return tf.nn.relu(margin)


class SequenceMLP(object):
    def __init__(self, config, phase='train'):
        self._phase = phase
        self._config = config
        self._vocab_size = config.vocab_size
        self._keep_prob = 0.5 if self._phase == 'train' else 1.0
        self._weight_decay = 1e-4
        self._word_embed_dim = config.word_embed_dim
        self._initializer = get_default_initializer()

        # list variables
        self._seq_inputs = self._att_mask = None
        self._mask_feed = self._att_feed = None
        self._word_embedding = None
        self._labels = None
        self._tot_loss = self._logits = None
        self.global_step = None
        self._feed_keys = None

    def add_logit_regulariser(self):
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('logits/weights')
        return tf.mul(self._weight_decay, tf.nn.l2_loss(weights))

    def build_inputs(self):
        self._seq_inputs = tf.placeholder(dtype=tf.int64, shape=[None, None])
        self._att_feed = tf.placeholder(dtype=tf.float32, shape=[None, None])  # NxT
        self._labels = tf.placeholder(dtype=tf.float32, shape=[None])
        self._feed_keys = [self._seq_inputs, self._att_feed]
        if self._phase == 'train':
            self._feed_keys.append(self._labels)
        # expand dim
        self._att_mask = tf.expand_dims(self._att_feed, 2)  # NxTx1

    def fill_feed_dict(self, inputs):
        return {k: v for (k, v) in zip(self._feed_keys, inputs)}

    def build_embedding(self):
        with tf.variable_scope('word_embedding'):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self._vocab_size, self._word_embed_dim],
                initializer=self._initializer)
            self._word_embedding = tf.nn.embedding_lookup(embedding_map, self._seq_inputs)

    def build_model(self):
        with tf.variable_scope('h'):
            h_unfold = tf.mul(self._word_embedding, self._att_mask)
            h = tf.nn.tanh(tf.reduce_sum(h_unfold, reduction_indices=1))
        h_drop = tf.nn.dropout(h, self._keep_prob)
        # compute outputs
        self._logits = tf.contrib.layers.fully_connected(inputs=h_drop,
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
        self.build_embedding()
        self.build_model()

    @property
    def tot_loss(self):
        return self._tot_loss

    @property
    def prob(self):
        return self._logits


def _process_datum(datum, thresh=3000):
    seq_index = datum['quest_seq']
    ids = seq_index[0]
    rm_entry = ids > thresh
    seq_index[:, rm_entry] = thresh
    att_mask = datum['perplex']
    att_mask[:, rm_entry] = 0
    seq_len = att_mask.shape[1] - rm_entry.sum()
    att_mask = att_mask / seq_len
    # get label
    label = datum['label']
    return seq_index, att_mask, label


def train():
    from util import unpickle
    import os
    import numpy.random as nr

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Create model
    max_iter = 1000000
    config = MLPConfig()
    model = SequenceMLP(config, phase='train')
    model.build()

    loss = model.tot_loss
    # global_step = model.global_stepgg
    train_op = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    dataset = unpickle('data/rescore_trainval.pkl')
    import pdb
    pdb.set_trace()
    saver = tf.train.Saver(max_to_keep=5)
    num = len(dataset)
    for i in range(max_iter):
        sample_id = nr.randint(0, num)
        datum = dataset[sample_id]
        seq_index, att_mask, label = _process_datum(datum)
        feed_dict = model.fill_feed_dict([seq_index,
                                          att_mask, label])
        _, obj = sess.run([train_op, model.tot_loss], feed_dict=feed_dict)

        if i % 100 == 0:
            tf.logging.info('Iteration %d, loss=%0.2f' % (i, obj))

        if i % 5000 == 0:
            saver.save(sess, os.path.join(FLAGS.train_dir, 'model-%d.ckpt' % i))


def test():
    from util import unpickle
    import json
    from inference_utils.question_generator_util import SentenceGenerator
    from w2v_answer_encoder import MultiChoiceQuestionManger

    config = MLPConfig()
    model = SequenceMLP(config, phase='test')
    model.build()
    prob = model.prob

    # Load vocabulary
    to_sentence = SentenceGenerator(trainset='trainval')
    # create multiple choice question manger
    mc_manager = MultiChoiceQuestionManger(subset='trainval',
                                           answer_coding='sequence')


    sess = tf.Session()
    # Load model
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    checkpoint_path = ckpt.model_checkpoint_path
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # get data
    result = []
    dataset = unpickle('data/rescore_dev.pkl')
    for itr, datum in enumerate(dataset):
        seq_index, att_mask, label = _process_datum(datum)
        quest_id = datum['quest_id']
        quest = seq_index[0].tolist()
        feed_dict = model.fill_feed_dict([seq_index, att_mask])
        scores = sess.run(prob, feed_dict=feed_dict)
        idx = scores.argmax()
        # parse question and answer
        question = to_sentence.index_to_question([0]+quest)
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
    res_file = 'result/rescore_dev_dev.json'
    json.dump(result, open(res_file, 'w'))
    from vqa_eval import evaluate_model
    acc = evaluate_model(res_file, quest_ids)
    print ('Over all accuarcy: %0.2f' % acc)


if __name__ == '__main__':
    train()
