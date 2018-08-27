import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import mlb, _soft_attention_pool, _spatial_softmax, _get_tensor_shape


def _soft_attention(im, ctx, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        fv = _soft_attention_pool(im, im_ctx)
    # fv = tf.expand_dims(fv, 1)
    return fv


def semantic_attention(attr, quest_embed, embed_dim, keep_prob, scope='SemAtt'):
    with tf.variable_scope(scope):
        aq_embed = mlb(attr, quest_embed, embed_dim, keep_prob,
                       scope='gates')
        gates = slim.fully_connected(aq_embed, 1, activation_fn=tf.nn.sigmoid,
                                     scope='gates')
        # apply gates
        gated_attr = tf.mul(attr, gates)
    return gated_attr

# def _default_conv2d(inputs, num_filters, scope=''):
#     return slim.conv2d(inputs, num_filters, kernel_size=3,
#                        stride=2, padding='SAME', scope=scope)

def _default_conv2d(inputs, num_filters, keep_prob, scope=''):
    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, 256, kernel_size=1, scope='conv1_1')  # dim reduct
        net = slim.conv2d(net, num_filters, kernel_size=3, stride=2, scope='conv1_2')
        net = slim.dropout(net, keep_prob=keep_prob)
    return net


def _default_deconv2d(inputs, num_filters, keep_prob, scope=''):
    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, 256, kernel_size=1, scope='conv1_1')  # dim reduct
        net = slim.conv2d_transpose(net, num_filters, kernel_size=3, stride=2, scope='conv1_2')
        # net = slim.dropout(net, keep_prob=keep_prob)
    return net


def compute_gates(g_im, ctx, embed_dim, num_outputs, keep_prob):
    # is_training = keep_prob != 1.0
    g_h = mlb(g_im, ctx, embed_dim, keep_prob=keep_prob, scope='gate')
    g_logits = slim.fully_connected(g_h, num_outputs, activation_fn=None,
                                    scope='g_logits')
    return tf.nn.softmax(g_logits)


def multiscale_attention(net, ctx, embed_dim, keep_prob, ms_kernel_channels,
                         ms_embed_dim=256):
    v_basis = []
    # add scale 0
    v = _soft_attention(net, ctx, embed_dim, keep_prob=keep_prob, scope='s0')
    v_basis.append(v)

    # add other scales
    end_point = net
    for i, chn in enumerate(ms_kernel_channels):
        s_scope = 'scale_%d' % (i + 1)
        end_point = _default_conv2d(end_point, chn, keep_prob, scope=s_scope)
        v = _soft_attention(end_point, ctx, ms_embed_dim, keep_prob=keep_prob,
                            scope=s_scope)
        v_basis.append(v)
    v_basis = tf.concat(1, v_basis)
    return v_basis

    # # compute gate
    # g_im = tf.reduce_max(net, reduction_indices=[1, 2])
    # gates = compute_gates(g_im, ctx, gate_embed_dim, len(ms_kernel_channels) + 1, keep_prob)
    #
    # # apply gates
    # gates = tf.expand_dims(gates, 2)
    # gated_v = tf.reduce_sum(v_basis*gates, reduction_indices=1)
    # return gated_v


def _scale_specific_vq_prediction(net, ctx, embed_dim, num_ans, keep_prob, scope, expand_dim=True):
    with tf.variable_scope(scope):
        v = _soft_attention(net, ctx, embed_dim, keep_prob=keep_prob)
        pre_logits = mlb(v, ctx, embed_dim, keep_prob, scope='pre_logits')
        logits = slim.fully_connected(pre_logits, num_ans, activation_fn=None, scope='logits')
        if expand_dim:
            return tf.expand_dims(logits, 1)
        else:
            return logits


def gated_multiscale_attention(net, ctx, embed_dim, num_ans, keep_prob,
                               ms_kernel_channels, ms_embed_dim=256,
                               gate_embed_dim=256, use_gate=True):
    basis = []
    # add scale 0
    v = _scale_specific_vq_prediction(net, ctx, embed_dim, num_ans, keep_prob=keep_prob, scope='scale_0')
    basis.append(v)

    # add other scales
    end_point = net
    for i, chn in enumerate(ms_kernel_channels):
        s_scope = 'scale_%d' % (i + 1)
        end_point = _default_conv2d(end_point, chn, keep_prob, scope=s_scope)
        v = _scale_specific_vq_prediction(end_point, ctx, ms_embed_dim, num_ans,
                                          keep_prob=keep_prob, scope=s_scope)
        basis.append(v)
    basis = tf.concat(1, basis)

    if use_gate:
        # compute gate
        g_im = tf.reduce_max(net, reduction_indices=[1, 2])
        gates = compute_gates(g_im, ctx, gate_embed_dim, len(ms_kernel_channels) + 1, keep_prob)

        # apply gates
        gates = tf.expand_dims(gates, 2)
        gated_logits = tf.reduce_sum(basis * gates, reduction_indices=1)
        return gated_logits
    else:
        return tf.reduce_mean(basis, reduction_indices=1)


def upsample_attention(net, ctx, embed_dim, num_ans, keep_prob):
    # add other scales
    end_point = net
    s_scope = 'ups_0'
    end_point = _default_deconv2d(end_point, 1024, keep_prob, scope=s_scope)
    logits = _scale_specific_vq_prediction(end_point, ctx, embed_dim, num_ans,
                                           keep_prob, scope=s_scope, expand_dim=False)
    return logits


def _low_rank_attention_pool(im, im_ctx, num_rank=32):
    im_chn = _get_tensor_shape(im)[-1]
    att_logits = slim.conv2d(im_ctx, num_rank, 1,
                             activation_fn=None,
                             scope='att_logits')
    prob = _spatial_softmax(att_logits)
    ups_att = slim.conv2d(prob, im_chn, kernel_size=1,
                          activation_fn=tf.nn.sigmoid,
                          scope='ups_att')
    gated = tf.mul(im, ups_att, name='gate')
    fv = tf.reduce_sum(gated, reduction_indices=[1, 2])
    fv = tf.nn.l2_normalize(fv, dim=1)
    return fv


def low_rank_attention(im, ctx, embed_dim, num_rank,
                       keep_prob, scope='LR_att'):
    scope = scope or "LR_att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        fv = _low_rank_attention_pool(im, im_ctx, num_rank)
    return fv


if __name__ == '__main__':
    import numpy.random as nr
    import numpy as np
    im = tf.constant(nr.rand(10, 14, 14, 2048), dtype=tf.float32)
    im_ctx = tf.constant(nr.rand(10, 14, 14, 1200), dtype=tf.float32)
    v = _low_rank_attention_pool(im, im_ctx, 64)
    v = _low_rank_attention_pool(im, im_ctx, 64)

