import tensorflow as tf
import tensorflow.contrib.slim as slim


# =========================== VQA losses ===========================
def _gather_prob(values, cols):
    num = tf.shape(values)[0]
    chns = tf.shape(values)[1]
    values = tf.reshape(values, [-1])
    indices = tf.range(start=0, limit=num,
                       dtype=tf.int32) * chns + cols
    return tf.gather(values, indices)


def _ranking_loss(s_pos, s_neg, delta):
    margin = s_neg + delta - s_pos
    losses = tf.nn.relu(margin)
    return losses
    # return tf.reduce_mean(losses * 0.1, name='RK')


def _split_tensor(values, num_splits, axis):
    if tf.__version__ == '0.12.0':
        return tf.split(axis, num_splits, values)
    else:
        return tf.split(values, num_splits, axis=axis)


def contrastive_ranking_loss(prob, labels, margin=0.3, loss_weight=1.0):
    """
    Pairwise ranking loss, it ensures the confidence the target class of the
    positive classes are larger than the negative class with a margin. The data
    pass to this class can be split into two groups: the former half batch size
    samples and the latter half are mutually contrastive, i.e., having the same
    question but answers(labels) differ.
    """
    # slice prob and labels
    probs_pos, probs_cst = _split_tensor(prob, 2, axis=0)
    t_pos, t_cst = _split_tensor(labels, 2, axis=0)
    # default
    d_pos = _gather_prob(probs_pos, t_pos)
    d_neg = _gather_prob(probs_cst, t_pos)
    d_loss = _ranking_loss(d_pos, d_neg, margin)
    # contrastive
    c_pos = _gather_prob(probs_cst, t_cst)
    c_neg = _gather_prob(probs_pos, t_cst)
    c_loss = _ranking_loss(c_pos, c_neg, margin)
    loss = tf.multiply(c_loss + d_loss, loss_weight, name='ranking_loss')
    slim.losses.add_loss(loss)
    return loss


def contrastive_question_ranking_loss(prob, labels, margin=0.3,
                                      loss_weight=1.0):
    # slice prob and labels
    probs_pos, probs_cst = _split_tensor(prob, 2, axis=0)
    t_pos, t_cst = _split_tensor(labels, 2, axis=0)
    # default
    d_pos = _gather_prob(probs_pos, t_pos)
    d_neg = _gather_prob(probs_cst, t_pos)
    d_loss = _ranking_loss(d_pos, d_neg, margin)
    # contrastive
    losses = tf.multiply(d_loss, loss_weight)
    loss = tf.reduce_mean(losses, name='ranking_loss')
    tf.losses.add_loss(loss)
    return loss


def multidomain_classification_loss(logits, labels, loss_weight=1.0):
    # slice prob and labels
    bin_logits = _gather_prob(logits, labels)  # logits
    bin_logits = tf.expand_dims(bin_logits, 1)
    batch_size = tf.cast(tf.shape(logits)[0], tf.int32)
    half_batch_size = tf.div(batch_size, tf.constant(2, dtype=tf.int32))
    bin_labels = tf.concat([tf.ones([half_batch_size, 1]),
                            tf.zeros([half_batch_size, 1])],
                           axis=0)
    tf.losses.sigmoid_cross_entropy(multi_class_labels=bin_labels,
                                    logits=bin_logits,
                                    weights=loss_weight,
                                    scope='domain_loss')


def conditional_cross_entropy_loss(logits, mask, loss_weight=1.0):
    # slice prob and labels
    prob = tf.nn.softmax(logits)
    weights = prob * tf.expand_dims(mask, 1) * loss_weight
    log_prob = tf.nn.log_softmax(logits, name='log_prob')
    losses = tf.reduce_sum(log_prob * weights, axis=1)  # maximise entropy, so drop the negative sign
    loss = tf.reduce_mean(losses, name='cxent')
    tf.losses.add_loss(loss)


def test_cst_ranking_loss():
    import numpy as np
    prob_feed = np.random.rand(8, 4)
    labels_feed = np.random.randint(0, 4, size=(8,), dtype=np.int32)

    # prob = tf.placeholder(tf.float32, shape=[None, 4])
    # labels = tf.placeholder(tf.int32, shape=[None])

    prob = tf.constant(prob_feed, dtype=tf.float32)
    labels = tf.constant(labels_feed, dtype=tf.int32)

    loss = contrastive_ranking_loss(prob, labels)

    loss_val = loss.eval(feed_dict={prob: prob_feed, labels: labels_feed})
    print(loss_val)


# ========================= Inverse VQA losses =========================
# def stride_cross_entropy_loss(logits, targets, mask, k):
#     logits = _stride_slice_and_flatten(logits, k)
#     targets = _stride_slice_and_flatten(targets, k)
#     mask = _stride_slice_and_flatten(mask, k)
#     # compute loss
#     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
#                                                             logits=logits)
#     total_loss = tf.div(tf.reduce_sum(losses * mask), tf.reduce_sum(mask),
#                         name='XE')
#     slim.losses.add_loss(total_loss)

def add_softmax_loss(neg_logprobs, mask):
    '''
    :param logits: tensor of shape (NxRxT), where R is the number of replicates
    :param targets: tensor of shape (NxRxT)
    :param mask: tensor of shape (NxRxT)
    :return:
    '''
    num_replica = tf.shape(neg_logprobs)[1]

    # ------------ compute cross entropy loss ------------
    pos_logprob = neg_logprobs[:, ::num_replica, :]
    pos_mask = mask[:, ::num_replica, :]
    xe_loss = tf.div(tf.reduce_sum(pos_logprob),
                     tf.reduce_sum(pos_mask), name='XE')
    slim.losses.add_loss(xe_loss)

    # -------------- answer ranking --------------
    with tf.variable_scope('Softmax'):
        temp = tf.get_variable('T', shape=[1], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=10.,
                                                                   dtype=tf.float32),
                               trainable=True)
        bias = tf.get_variable('B', shape=[1], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.,
                                                                   dtype=tf.float32),
                               trainable=True)
    logprobs = tf.negative(neg_logprobs)
    # log_p_sum = tf.reduce_sum(logprobs, axis=2)
    p_logits = tf.exp(tf.reduce_sum(logprobs, axis=2) + temp)  # NxR
    p_logits = p_logits + bias

    # softmax loss
    batch_size = tf.shape(neg_logprobs)[0]
    gt = tf.zeros([batch_size], dtype=tf.int32)  # the first one is always the gt
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt,
                                                            logits=p_logits)
    softmax_loss = tf.reduce_mean(0.1 * losses, name='CST')
    slim.losses.add_loss(softmax_loss)
    return p_logits


def add_pairwise_loss(neg_logprobs, mask, delta):
    '''
    :param logits: tensor of shape (NxRxT), where R is the number of replicates
    :param targets: tensor of shape (NxRxT)
    :param mask: tensor of shape (NxRxT)
    :return:
    '''
    num_replica = tf.shape(neg_logprobs)[1]

    # ------------ compute cross entropy loss ------------
    pos_logprob = neg_logprobs[:, ::num_replica, :]
    pos_mask = mask[:, ::num_replica, :]
    xe_loss = tf.div(tf.reduce_sum(pos_logprob * pos_mask),
                     tf.reduce_sum(pos_mask), name='XE')
    slim.losses.add_loss(xe_loss)

    # -------------- answer ranking --------------
    logprobs = tf.negative(neg_logprobs)
    logits = tf.reduce_sum(logprobs, axis=2)
    p_logits = tf.reshape(logits[:, 0], [-1, 1])
    n_logits = logits[:, 1:]
    rank_loss = _ranking_loss(p_logits, n_logits, delta)
    slim.losses.add_loss(rank_loss)
    return logits


def _stride_slice_and_flatten(t, stride, idx=0):
    return tf.reshape(t[idx::stride, :], [-1])


if __name__ == '__main__':
    test_cst_ranking_loss()
