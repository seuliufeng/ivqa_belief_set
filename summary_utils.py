import tensorflow as tf


def _get_tensor_rank(x):
    s = x.get_shape().as_list()
    return len(s)


def add_vector_summaries(tags, var):
    with tf.name_scope(tags):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_gradient_summaries(tags, ys, xs):
    grad = tf.gradients(ys, xs)[0]
    if grad is None:
        return
    grad = tf.abs(grad)  # we don't care about the sign
    with tf.name_scope(tags):
        mean = tf.reduce_mean(grad)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(grad - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(grad))
        tf.summary.scalar('min', tf.reduce_min(grad))
        tf.summary.histogram('histogram', grad)


def add_accuracy_summary(tags, labels, logits):
    pred_labels = tf.cast(tf.argmax(logits, 1), labels.dtype)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels),
                                      tf.float32))
    tf.summary.scalar(tags, accuracy)


def add_count_summaries(tags, mask, compute_mask=False):
    """
    visualise the mean activation over the samples
    inputs:
    mask: a Nxd tensor of gate activations, where N is
    the number of samples
    """
    if compute_mask:
        mask = tf.stop_gradient(mask)
        tmp, _ = tf.nn.top_k(mask, k=1)
        mask = tf.gradients(tmp, mask)[0]
    with tf.name_scope(tags):
        counts = tf.reduce_sum(mask, reduction_indices=0)
        tf.summary.histogram('histogram', counts)


def add_attention_map_summary(tags, att_map):
    if _get_tensor_rank(att_map) == 3:
        att_map = tf.expand_dims(att_map, 3)
    att_map = tf.image.resize_bilinear(att_map, [256, 256])
    tf.summary.image(tags, att_map, max_outputs=1)

