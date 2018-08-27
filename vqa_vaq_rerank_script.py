import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from util import load_hdf5


def load_dataset(subset='dev', split=None):
    # load VQA results
    print('Dataset: Loading VQA results...')
    vqa_file = 'data/vqa-advrl_vqa_score2000_%s.hdf5' % subset
    print('Loading %s' % vqa_file)
    d = load_hdf5(vqa_file)
    vqa_ids = d['quest_ids']
    vqa_scores = d['confidence'][:, :2000]
    gt_labels = d['labels']

    # Load SPLIT 0
    print('Dataset: Loading VAQ results <SPLIT 0>...')
    vaq_file = 'data/vaq-van_vaq_cond_score1000_%s.hdf5' % subset
    d = load_hdf5(vaq_file)
    vaq_loss = d['nll']
    vaq_ids = d['quest_ids']
    quest_id2vaq_index = {qid: i for i, qid in enumerate(vaq_ids)}
    new_index = [quest_id2vaq_index[qid] for qid in vqa_ids]
    vaq_loss_0 = vaq_loss[new_index]

    # Load SPLIT 1
    print('Dataset: Loading VAQ results <SPLIT 1>...')
    vaq_file = 'data/vaq-van_vaq_cond_score1000-2000_%s.hdf5' % subset
    d = load_hdf5(vaq_file)
    vaq_loss = d['nll']
    vaq_ids = d['quest_ids']
    quest_id2vaq_index = {qid: i for i, qid in enumerate(vaq_ids)}
    new_index = [quest_id2vaq_index[qid] for qid in vqa_ids]
    vaq_loss_1 = vaq_loss[new_index]

    vaq_loss = np.concatenate([vaq_loss_0, vaq_loss_1], 1)

    # Split dataset because training set has over-fitted
    if subset == 'train' or split is None:
        return vqa_scores, vaq_loss, gt_labels, vqa_ids
    else:
        num = gt_labels.size
        num_train = int(num * 0.7)
        if split == 0:
            print('Loading training split')
            vqa_scores = vqa_scores[:num_train, :]
            vaq_loss = vaq_loss[:num_train, :]
            gt_labels = gt_labels[:num_train]
        else:
            print('Loading testing split')
            vqa_scores = vqa_scores[num_train:, :]
            vaq_loss = vaq_loss[num_train:, :]
            gt_labels = gt_labels[num_train:]
        return vqa_scores, vaq_loss, gt_labels, vqa_ids


def check_vqa_result():
    vqa_scores, vaq_loss, gt_labels = load_dataset('dev')
    print('Test on VQA:')
    test_accuracy(vqa_scores, gt_labels)
    #
    print('Test on VAQ:')
    test_accuracy(-vaq_loss, gt_labels)


def test_accuracy(scores, gt_labels):
    n_tot = gt_labels.size
    valid = gt_labels != 2000
    scores = scores[valid, :]
    gt_labels = gt_labels[valid]
    pred_labels = scores.argmax(axis=1)
    accuracy = 100.0 * np.sum(pred_labels == gt_labels) / float(gt_labels.size)
    print('Eval %d/%d samples, accuracy: %0.2f' % (gt_labels.size,
                                                   n_tot, accuracy))
    return accuracy


# ==================== SYMBOL FN =========================

def scale_inputs(input, scope):
    num_chn = input.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        scaler = tf.get_variable(shape=[1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(value=1.,
                                                                     dtype=tf.float32),
                                 name='scalar')
        return input * scaler


def create_variable(var_shape, scope):
    with tf.variable_scope(scope):
        var = tf.get_variable(shape=var_shape, dtype=tf.float32,
                              initializer=tf.constant_initializer(value=1.,
                                                                  dtype=tf.float32),
                              name='scalar')
        return var


def build_vaq_feature_preprocesser(vaq_feat, proc_type='softmax'):
    with tf.variable_scope('VAQProc'):
        if proc_type == 'softmax':
            log_likehood = -vaq_feat
            # scaler = create_variable(var_shape=[1, 1000], scope='scalar')
            scaler = create_variable(var_shape=[1], scope='scalar')
            log_likehood = log_likehood * scaler
            return tf.nn.softmax(log_likehood)
        elif proc_type == 'exp':
            log_likehood = -vaq_feat
            # scaler = 5.0
            scaler = create_variable(var_shape=[1], scope='scalar')
            log_likehood = log_likehood * scaler
            return tf.exp(log_likehood)
        else:
            return tf.nn.l2_normalize(vaq_feat, 1)


def _multi_class_cross_entropy_loss(pred, labels, EPS=1e-12):
    num_classes = 2000
    labels = slim.one_hot_encoding(labels, num_classes)
    losses = - labels * tf.log(pred + EPS)
    return tf.reduce_mean(losses) * num_classes


def _multi_class_l2_loss(pred, labels, EPS=1e-12):
    num_classes = 2000
    labels = slim.one_hot_encoding(labels, num_classes)
    losses = tf.squared_difference(pred, labels)
    loss = tf.reduce_mean(losses) * num_classes
    slim.losses.add_loss(loss)
    return loss


def _l1_normalize(input):
    return tf.div(input, tf.expand_dims(tf.reduce_sum(input, 1), 1))  # l1 normalize


def apply_vqa_mask(vaq_pred, vqa_pred, k=5):
    tmp, _ = tf.nn.top_k(vqa_pred, k=k, sorted=False)
    mask = tf.stop_gradient(tf.gradients(tmp, vqa_pred)[0])
    return vaq_pred * mask, mask


def build_classfication_net(vqa_feat, vaq_feat, labels, keep_prob):
    # process vaq prediction
    with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(5e-4)):
        inputs = build_vaq_feature_preprocesser(vaq_feat, proc_type='other')
        h_0 = slim.fully_connected(inputs, 512, scope='hidden')
        h_0 = slim.dropout(h_0, keep_prob=keep_prob)
        vaq_pred = slim.fully_connected(h_0, 2000,
                                        activation_fn=None,
                                        scope='vaq_classifier')
    loss = slim.losses.sparse_softmax_cross_entropy(vaq_pred, labels)
    return vaq_pred, loss


def learn_combination_weights(vqa_feat, vaq_pred, labels):
    masked_pred = apply_vqa_mask(tf.nn.softmax(vaq_pred), vqa_feat, k=5)
    fused = vqa_feat * masked_pred
    loss = _multi_class_l2_loss(fused, labels)
    return fused, loss


def build_classification_net_v0(vqa_pred, vaq_loss, labels):
    vaq_lh = build_vaq_feature_preprocesser(vaq_loss, proc_type='exp')
    vaq_lh, mask = apply_vqa_mask(vaq_lh, vqa_pred, k=5)
    vaq_mask = _l1_normalize(vaq_lh)
    pred = vaq_mask * vqa_pred
    pred = _l1_normalize(pred)
    loss = _multi_class_l2_loss(pred, labels)
    return pred, loss, mask



    # # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(vaq_pred, labels))
    #
    # if apply_mask:
    #     vaq_pred = apply_vqa_mask(tf.exp(vaq_pred), vqa_feat, k=100)
    # else:
    #     vaq_pred = tf.exp(vaq_pred)
    # # vaq_pred = tf.nn.softmax(vaq_pred)
    # vaq_pred = _l1_normalize(vaq_pred)
    #
    # # merge features
    # # preds = _l1_normalize(vaq_pred)  # l1 normalize
    # # gate = create_variable([1, 2], scope='gate')
    # # gate = tf.nn.softmax(gate)
    # # gate_0, gate_1 = tf.split(1, 2, gate)
    # # fused_pred = gate_0 * vaq_pred + gate_1 * vqa_feat
    # fused_pred = vaq_pred
    # # fused_pred = 0.5 * (vaq_pred + vqa_feat)
    #
    # # compute loss
    # loss_vaq = _multi_class_cross_entropy_loss(vaq_pred, labels)
    # # loss_fused = _multi_class_l2_loss(fused_pred, labels)
    # # loss = loss_vaq + loss_fused
    # loss = loss_vaq
    # # loss = loss_fused
    # return fused_pred, loss


def assgin_T(sess, T):
    with tf.variable_scope('VAQProc/scalar', reuse=True):
        var_name = 'scalar'
        var = tf.get_variable(var_name)
        var_shape = var.get_shape().as_list()
        T = np.array(T).reshape(var_shape)
        sess.run(var.assign(T))


def train():
    train_set = 'trainval'
    test_set = 'dev'
    num_iters = 100000
    batch_size = 256

    # slice vaq feature maybe
    max_vaq_dim = 2000

    # build graph
    vqa_feed = tf.placeholder(tf.float32, shape=[None, 2000])
    vaq_feed = tf.placeholder(tf.float32, shape=[None, max_vaq_dim])
    label_feed = tf.placeholder(tf.int32, shape=[None])
    keep_prob = tf.placeholder(tf.float32, shape=None)
    vaq_pred, loss, mask = build_classification_net_v0(vqa_feed,
                                                       vaq_feed,
                                                       label_feed)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # build finetune step
    # fused_pred, ft_loss = learn_combination_weights(vqa_feed, vaq_pred, label_feed)
    # finetune_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(ft_loss)

    # start session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # start training
    # vqa_train, vaq_train, gt_train = load_dataset(train_set)
    # num = gt_train.size
    # index = np.arange(num)
    # for i in range(num_iters):
    #     idx = np.random.choice(index, batch_size)
    #     b_vqa_score = vqa_train[idx, :]
    #     b_vaq_score = vaq_train[idx, :max_vaq_dim]
    #     b_gt_label = gt_train[idx]
    #     _, b_loss = sess.run([train_step, loss], feed_dict={vqa_feed: b_vqa_score,
    #                                                         vaq_feed: b_vaq_score,
    #                                                         label_feed: b_gt_label,
    #                                                         keep_prob: 0.7})
    #     if i % 1000 == 0:
    #         print('Training: iter %d/%d, loss %0.3f' % (i, num_iters, b_loss))
    #
    # # Test on training set
    # vqa_test, vaq_test, gt_test = vqa_train, vaq_train, gt_train
    # num = gt_train.size
    # num_batches = int(np.ceil(num / float(batch_size)))
    #
    # v_preds = []
    # for i in range(num_batches):
    #     batch_beg = i * batch_size
    #     batch_end = min(num, (i + 1) * batch_size)
    #     # slice testing data
    #     b_vqa_score = vqa_test[batch_beg:batch_end, :]
    #     b_vaq_score = vaq_test[batch_beg:batch_end, :max_vaq_dim]
    #     b_pred = sess.run(vaq_pred, feed_dict={vqa_feed: b_vqa_score,
    #                                            vaq_feed: b_vaq_score,
    #                                            keep_prob: 1.0})
    #     v_preds.append(b_pred)
    #     if i % 1000 == 0:
    #         print('Testing: iter %d/%d' % (i, num_batches))
    #
    # v_preds = np.concatenate(v_preds, axis=0)
    # print('Test on Training split:')
    # test_accuracy(v_preds, gt_test)

    # # Finetune on dev set split 0
    # vqa_train, vaq_train, gt_train = load_dataset('dev', split=0)
    # num = gt_train.size
    # index = np.arange(num)
    # for i in range(100000):
    #     idx = np.random.choice(index, batch_size)
    #     b_vqa_score = vqa_train[idx, :]
    #     b_vaq_score = vaq_train[idx, :max_vaq_dim]
    #     b_gt_label = gt_train[idx]
    #     _, b_loss = sess.run([train_step, loss], feed_dict={vqa_feed: b_vqa_score,
    #                                                         vaq_feed: b_vaq_score,
    #                                                         label_feed: b_gt_label})
    #     if i % 1000 == 0:
    #         print('Training: iter %d/%d, loss %0.3f' % (i, num_iters, b_loss))
    #

    # Test on test set
    vqa_test, vaq_test, gt_test, quest_ids = load_dataset('dev')
    num = gt_test.size
    num_batches = int(np.ceil(num / float(batch_size)))

    print('\n============================')
    print('Before re-ranking:')
    test_accuracy(vqa_test, gt_test)

    # values = np.linspace(0, 4, num=80, dtype=np.float32)
    values = [2.025]
    for tem in values:
        assgin_T(sess, tem)
        v_preds = []
        for i in range(num_batches):
            batch_beg = i * batch_size
            batch_end = min(num, (i + 1) * batch_size)
            # slice testing data
            b_vqa_score = vqa_test[batch_beg:batch_end, :]
            b_vaq_score = vaq_test[batch_beg:batch_end, :max_vaq_dim]
            b_pred, b_mask = sess.run([vaq_pred, mask], feed_dict={vqa_feed: b_vqa_score,
                                                                   vaq_feed: b_vaq_score})
            v_preds.append(b_pred)
            # if i % 1000 == 0:
            #     print('Testing: iter %d/%d' % (i, num_batches))

        v_preds = np.concatenate(v_preds, axis=0)
        print('\n============== T=%0.3f ==============' % tem)
        print('Test on Testing split:')
        test_accuracy(v_preds, gt_test)

        # generate result and test
        from inference_utils.question_generator_util import SentenceGenerator
        import json
        to_sentence = SentenceGenerator(trainset='trainval')
        # answer_index = v_preds.argmax(axis=1)
        answer_index = vqa_test.argmax(axis=1)
        result = []
        for (ans_id, quest_id) in zip(answer_index, quest_ids):
            ans = to_sentence.index_to_top_answer(ans_id)
            result.append({u'answer': ans, u'question_id': quest_id})
        # save results
        tf.logging.info('Saving results')
        res_file = 'result/tmp.json'
        json.dump(result, open(res_file, 'w'))
        from vqa_eval import evaluate_model
        evaluate_model(res_file, quest_ids)


if __name__ == '__main__':
    # check_vqa_result()
    train()
