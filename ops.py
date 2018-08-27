import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import os
from summary_utils import *

# from compact_bilinear_pooling import compact_bilinear_pooling_layer

EPS = 1e-16


def get_variable_scope(x):
    return os.path.dirname(x.name)


def compute_squared_cv(x, EPS=1e-12):
    """
    compute the squared coefficient of variation
    """
    mean = tf.reduce_mean(x)
    variance = tf.reduce_mean(tf.square(x - mean))
    return tf.div(variance, tf.square(mean) + EPS)


def prelu(x, alpha=1.0):
    scope = get_variable_scope(x) or ''
    with tf.variable_scope(scope + '/PReLU'):
        alphas = tf.get_variable('alpha', _get_tensor_shape(x)[-1:],
                                 initializer=tf.constant_initializer(alpha),
                                 dtype=tf.float32)
        x = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5

        return x


def prelu2(x, alpha=1.0):
    with tf.variable_scope('PReLU'):
        alphas = tf.get_variable('alpha', _get_tensor_shape(x)[-1:],
                                 initializer=tf.constant_initializer(alpha),
                                 dtype=tf.float32)
        x = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5

        return x


def signed_sqrt(x, normalise=True):
    y = tf.mul(tf.sign(x), tf.sqrt(tf.abs(x) + EPS))
    if normalise:
        y = tf.nn.l2_normalize(y, dim=1)
    return y


def expand_x_y_dims(f):
    return tf.expand_dims(tf.expand_dims(f, 1), 1)


def _get_tensor_shape(x):
    s = x.get_shape().as_list()
    return [i if i is not None else -1 for i in s]


def _get_tensor_rank(x):
    s = x.get_shape().as_list()
    return len(s)


def _is_4d_tensor(x):
    return len(_get_tensor_shape(x)) == 4


def mlb(im, ctx, embed_dim, keep_prob, scope=""):
    scope = scope or "mbp"
    with tf.variable_scope(scope):
        ctx_emb = slim.fully_connected(ctx, embed_dim,
                                       activation_fn=tf.nn.tanh,
                                       scope='ctx_emb')
        # ctx_emb = slim.dropout(ctx_emb, keep_prob=keep_prob)
        if _is_4d_tensor(im):
            ctx_emb = expand_x_y_dims(ctx_emb)
            im_emb = slim.conv2d(im, embed_dim, [1, 1],
                                 activation_fn=tf.nn.tanh,
                                 scope='im_emb')
        else:
            im_emb = slim.fully_connected(im, embed_dim,
                                          activation_fn=tf.nn.tanh,
                                          scope='im_emb')
        # im_emb = slim.dropout(im_emb, keep_prob=keep_prob)
        embed = im_emb * ctx_emb
        embed = slim.dropout(embed, keep_prob=keep_prob)
    return embed


def concat_fusion(im, ctx, embed_dim, act_fn=tf.nn.relu):
    branch_0 = slim.fully_connected(ctx, embed_dim,
                                    activation_fn=None,
                                    scope='branch_0')
    branch_1 = slim.fully_connected(im, embed_dim,
                                    activation_fn=None,
                                    scope='branch_1')
    if act_fn is None:
        return branch_1 + branch_0
    else:
        return act_fn(branch_0 + branch_1)


def mm_conv_concat(im, ctx, embed_dim, keep_prob, scope=""):
    '''
    Multi-Modals fusion, embed to the same dimensionality and then concat
    '''
    scope = scope or "mbp"
    with tf.variable_scope(scope):
        ctx_emb = slim.fully_connected(ctx, embed_dim, scope='ctx_emb')
        assert (_is_4d_tensor(im))
        ctx_emb = expand_x_y_dims(ctx_emb)
        _, w, h, _ = _get_tensor_shape(im)
        ctx_emb = tf.tile(ctx_emb, [1, w, h, 1])
        im_emb = slim.conv2d(im, embed_dim, [1, 1], scope='im_emb')
        embed = tf.concat(concat_dim=3, values=[im_emb, ctx_emb])
        embed = slim.dropout(embed, keep_prob=keep_prob)
    return embed


def _normalize_channel(x):
    n_rank = _get_tensor_rank(x)
    return tf.nn.l2_normalize(x, n_rank - 1)


def _nd_inner_product(x, y):
    n_rank_x = _get_tensor_rank(x)
    n_rank_y = _get_tensor_rank(y)
    assert (n_rank_x == n_rank_y)
    m = tf.mul(x, y)
    return tf.reduce_sum(m, reduction_indices=n_rank_x - 1)


def david_pool(im, ctx, scope=""):
    scope = scope or "dpool"
    embed_dim = _get_tensor_shape(im)[-1]
    im = _normalize_channel(im)
    with tf.variable_scope(scope):
        ctx_emb = slim.fully_connected(ctx, embed_dim, scope='ctx_emb')
        ctx_emb = _normalize_channel(ctx_emb)
        if _is_4d_tensor(im):
            ctx_emb = expand_x_y_dims(ctx_emb)

        # compute inner product
        embed = _nd_inner_product(ctx_emb, im)
    embed = tf.expand_dims(embed, 3)
    return embed


def _spatial_softmax(fm):
    fm_shape = _get_tensor_shape(fm)
    n_grids = fm_shape[1] ** 2
    # transpose feature map
    fm = tf.transpose(fm, perm=[0, 3, 1, 2])
    t_fm_shape = _get_tensor_shape(fm)
    fm = tf.reshape(fm, shape=[-1, n_grids])
    # apply softmax
    prob = tf.nn.softmax(fm)
    # reshape back
    prob = tf.reshape(prob, shape=t_fm_shape)
    prob = tf.transpose(prob, perm=[0, 2, 3, 1])
    return prob


def _soft_attention_pool(im, im_ctx):
    att_logits = slim.conv2d(im_ctx, 1, [1, 1],
                             activation_fn=None,
                             scope='att_logits')
    att = _spatial_softmax(att_logits)
    im_att = att * im
    return tf.reduce_sum(im_att, axis=[1, 2]), att


def _sigmoid_attention_pool(im, im_ctx, norm=True):
    att_logits = slim.conv2d(im_ctx, 1, [1, 1],
                             activation_fn=tf.nn.sigmoid,
                             scope='att_logits')
    im_att = tf.mul(att_logits, im)
    v = tf.reduce_sum(im_att, reduction_indices=[1, 2])
    if norm:
        v = tf.nn.l2_normalize(v, dim=1)
    return v, att_logits


def _soft_attention_pool_with_map(im, im_ctx):
    att_logits = slim.conv2d(im_ctx, 1, [1, 1],
                             activation_fn=None,
                             scope='att_logits')
    att = _spatial_softmax(att_logits)
    im_att = att * im
    return tf.reduce_sum(im_att, reduction_indices=[1, 2]), att


def _spatial_aggregate(im, att_map, normalize=True):
    assert (_get_tensor_rank(im) == 4)
    assert (_get_tensor_rank(att_map) == 4)
    # expand dims
    im = tf.expand_dims(im, dim=3)  # Nx14x14x1x2048
    att_map = tf.expand_dims(att_map, dim=4)  # Nx14x14xCx1
    # pooling over features
    im_att = tf.mul(att_map, im)
    im_feat = tf.reduce_sum(im_att, reduction_indices=[1, 2])
    if normalize:
        im_feat = tf.nn.l2_normalize(im_feat, dim=-1)
    return im_feat


def _spatial_aggregate_static(im, att_map, normalize=True):
    assert (_get_tensor_rank(im) == 4)
    assert (_get_tensor_rank(att_map) == 3)
    # expand dims
    im = tf.expand_dims(im, dim=3)  # Nx14x14x1x2048
    att_map = tf.expand_dims(att_map, dim=0)  # Nx14x14xCx1
    att_map = tf.expand_dims(att_map, dim=4)  # Nx14x14xCx1
    # pooling over features
    im_att = tf.mul(att_map, im)
    im_feat = tf.reduce_sum(im_att, reduction_indices=[1, 2])
    if normalize:
        im_feat = tf.nn.l2_normalize(im_feat, dim=-1)
    return im_feat


def _create_attention_map(im_ctx, reg_fn=_spatial_softmax,
                          scope=''):
    s_logits = slim.conv2d(im_ctx, 1, [1, 1], activation_fn=None,
                           scope=scope)
    return reg_fn(s_logits)


def sqrt_nneg_tensor(x, EPS=1e-12):
    return tf.sqrt(x + EPS)


# def create_attention_units(im, ctx, embed_dim, keep_prob=1.0, unit_id=0):
#     scope = "AttUnits%d" % unit_id
#     with tf.variable_scope(scope):
#         im_ctx = mlb(im, ctx, embed_dim, keep_prob)
#         # soft attention
#         att_max = _create_attention_map(im_ctx, scope='max')
#         v_max = tf.expand_dims(_spatial_aggregate(im, att_max), 1)
#         # sigmoid attention
#         att_sum = _create_attention_map(im_ctx, reg_fn=tf.nn.sigmoid,
#                                         scope='sum')
#         v_sum = tf.expand_dims(_spatial_aggregate(im, att_sum), 1)
#     return tf.concat(1, [v_max, v_sum])


def create_attention_units(im, ctx, embed_dim, keep_prob=1.0, unit_id=0, use_sigmoid=False):
    scope = "AttUnits%d" % unit_id
    outputs = []
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        # soft attention
        att_max = _create_attention_map(im_ctx, scope='max')
        v_max = _spatial_aggregate(im, att_max, normalize=use_sigmoid)
        outputs.append(v_max)
        # sigmoid attention
        if use_sigmoid:
            att_sum = _create_attention_map(im_ctx, reg_fn=tf.nn.sigmoid,
                                            scope='sum')
            v_sum = tf.expand_dims(_spatial_aggregate(im, att_sum, normalize=use_sigmoid), 1)
            outputs.append(v_sum)
    return tf.concat(1, outputs)


def gated_attention(im, quest, task_coding, embed_dim, keep_prob=1.0,
                    num_units=4, use_sum=True, reuse=False,
                    scope="GatedAttention"):
    num_gates = num_units * (1 + use_sum)
    with tf.variable_scope(scope, reuse=reuse):
        # att_basis = [create_attention_units(im, quest, embed_dim, keep_prob,
        #                                     unit_id, use_sigmoid=use_sum)
        #              for unit_id in range(num_units)]
        # att_basis = tf.concat(concat_dim=1, values=att_basis)
        im_ctx = mlb(im, quest, embed_dim, keep_prob)
        # soft attention
        s_logits = slim.conv2d(im_ctx, num_units, [1, 1], activation_fn=None,
                               scope='sp_logits')
        # spatial softmax
        att_maps = _spatial_softmax(s_logits)
        # aggregate visual features
        att_basis = _spatial_aggregate(im, att_maps, normalize=False)
        # compute gate
        with tf.variable_scope('Gate'):
            im_ctx = tf.reduce_mean(im, reduction_indices=[1, 2])
            vq_ctx = tf.concat(concat_dim=1, values=[im_ctx, quest])
            vq_embed = slim.fully_connected(vq_ctx, 256, scope='vq_embed')
            task_embed = slim.fully_connected(task_coding, 64, scope='task_embed')
            gates = slim.fully_connected(tf.concat(1, [vq_embed, task_embed]),
                                         num_gates, activation_fn=None,
                                         scope='gate')
            gates = tf.expand_dims(slim.softmax(gates), 2)
        output = tf.reduce_sum(tf.mul(att_basis, gates), reduction_indices=1)
    return output


def debug_gated_attention():
    import numpy as np
    import numpy.random as nr
    use_sum_map = False
    im_arr = nr.rand(2, 14, 14, 2048)
    quest_arr = nr.rand(2, 1024)
    im = tf.convert_to_tensor(im_arr, dtype=tf.float32)
    quest = tf.convert_to_tensor(quest_arr, dtype=tf.float32)
    vqa_task = tf.constant([0, 1], dtype=tf.float32)
    vqa_task = tf.tile(tf.reshape(vqa_task, shape=[1, 2]), [2, 1])
    vaq_task = tf.constant([1, 0], dtype=tf.float32)
    vaq_task = tf.tile(tf.reshape(vaq_task, shape=[1, 2]), [2, 1])
    vqa_coding = gated_attention(im, quest, vqa_task, 512, use_sum=use_sum_map)
    vaq_coding = gated_attention(im, quest, vaq_task, 512, reuse=True, use_sum=use_sum_map)
    sess.run(tf.initialize_all_variables())
    print vqa_coding.eval()
    print vaq_coding.eval()


def moe_attention(im, quest, embed_dim, keep_prob=1.0,
                  num_units=4, reuse=False, scope="MoeAttention"):
    with tf.variable_scope(scope, reuse=reuse):
        im_ctx = mlb(im, quest, embed_dim, keep_prob)
        # soft attention
        s_logits = slim.conv2d(im_ctx, num_units, [1, 1], activation_fn=None,
                               scope='sp_logits')
        # spatial softmax
        att_maps = _spatial_softmax(s_logits)
        # aggregate visual features
        att_basis = _spatial_aggregate(im, att_maps, normalize=False)
        # compute gate
        with tf.variable_scope('Gate'):
            im_ctx = tf.reduce_mean(im, reduction_indices=[1, 2])
            vq_embed = mlb(im_ctx, quest, embed_dim / 4, keep_prob, 'gate_prelogit')
            # vq_ctx = tf.concat(concat_dim=1, values=[im_ctx, quest])
            # vq_embed = slim.fully_connected(vq_ctx, embed_dim, scope='vq_embed')
            # vq_embed = slim.dropout(vq_embed, keep_prob)
            gates = slim.fully_connected(vq_embed, num_units, activation_fn=None,
                                         scope='gate')
            gates = tf.expand_dims(slim.softmax(gates), 2)
        output = tf.reduce_sum(tf.mul(att_basis, gates), reduction_indices=1)
        return output


def build_attention_banks(im, quest, embed_dim, keep_prob=1.0,
                          num_units=20, scope="AttBank"):
    with tf.variable_scope(scope):
        im_ctx = mlb(im, quest, embed_dim, keep_prob)
        # soft attention
        s_logits = slim.conv2d(im_ctx, num_units, [1, 1], activation_fn=None,
                               scope='sp_logits')
        # spatial softmax
        att_maps = _spatial_softmax(s_logits)
        # aggregate visual features
        att_basis = _spatial_aggregate(im, att_maps, normalize=False)
    return att_basis, att_maps


def build_human_att40(im, update=False, scope="AttBank"):
    from scipy.io import loadmat
    d = loadmat('att_bank40.mat')
    att_maps = d['att_bank'].astype(np.float32)
    att_maps = np.transpose(att_maps, [1, 2, 0])  # swap channels to last dim
    with tf.variable_scope(scope):
        att_maps = tf.Variable(initial_value=att_maps,
                               trainable=update,
                               dtype=tf.float32)
        # aggregate visual features
        att_basis = _spatial_aggregate_static(im, att_maps,
                                              normalize=False)
    return att_basis, att_maps


def build_human_att100(im, update=False, scope="AttBank"):
    from scipy.io import loadmat
    d = loadmat('att_bank100.mat')
    att_maps = d['att_bank'].astype(np.float32)
    att_maps = np.transpose(att_maps, [1, 2, 0])  # swap channels to last dim
    with tf.variable_scope(scope):
        att_maps = tf.Variable(initial_value=att_maps,
                               trainable=update,
                               dtype=tf.float32)
        # aggregate visual features
        att_basis = _spatial_aggregate_static(im, att_maps,
                                              normalize=False)
    return att_basis, att_maps


def compute_att100_gated_features(att_banks, q_embed, embed_dim,
                                  keep_prob, scope='Att100_Gate'):
    with tf.variable_scope(scope):
        im_reduct = slim.fully_connected(att_banks, embed_dim,
                                         activation_fn=tf.nn.tanh,
                                         scope='im_reduct')
        q_reduct = slim.fully_connected(q_embed, embed_dim,
                                        activation_fn=tf.nn.tanh,
                                        scope='quest_reduct')
        # multi-modal pooling
        vq_embed = tf.mul(tf.expand_dims(q_reduct, 1), im_reduct)
        vq_embed = slim.dropout(vq_embed, keep_prob)
        # compute logits
        gates = slim.fully_connected(vq_embed, 1, activation_fn=None, scope='gates')
        gates = tf.nn.softmax(gates, 1)
        # visual pooling
        att_pool = tf.reduce_sum(tf.mul(gates, att_banks), reduction_indices=1)
    return att_pool


def compute_reinforce_gated_features(reinforcer, att_banks, q_embed,
                                     embed_dim, keep_prob, scope='Att100_Gate'):
    with tf.variable_scope(scope):
        im_reduct = slim.fully_connected(att_banks, embed_dim,
                                         activation_fn=tf.nn.tanh,
                                         scope='im_reduct')
        q_reduct = slim.fully_connected(q_embed, embed_dim,
                                        activation_fn=tf.nn.tanh,
                                        scope='quest_reduct')
        # multi-modal pooling
        vq_embed = tf.mul(tf.expand_dims(q_reduct, 1), im_reduct)
        vq_embed = slim.dropout(vq_embed, keep_prob)
        # compute logits
        state = slim.fully_connected(vq_embed, 1, activation_fn=None,
                                     scope='gates')
        state = tf.squeeze(state, squeeze_dims=[2])
        action = reinforcer(state, state)
        # summarize wrt state
        prob = tf.nn.softmax(state)
        tf.histogram_summary('action', prob)
        add_count_summaries('act_counts', prob, compute_mask=True)
        # visual pooling
        _, c, _ = _get_tensor_shape(att_banks)
        gates = tf.expand_dims(tf.one_hot(action, depth=c), 2)
        att_pool = tf.reduce_sum(tf.mul(gates, att_banks), reduction_indices=1)
    return att_pool


def compute_sparse_moe_gated_features(att_banks, q_embed, embed_dim, num_active,
                                      keep_prob, decay_factor=0.01, phase='train',
                                      scope='SparseMOEGate'):
    with tf.variable_scope(scope):
        im_reduct = slim.fully_connected(att_banks, embed_dim,
                                         activation_fn=tf.nn.tanh,
                                         scope='im_reduct')
        q_reduct = slim.fully_connected(q_embed, embed_dim,
                                        activation_fn=tf.nn.tanh,
                                        scope='quest_reduct')
        # multi-modal pooling
        vq_embed = tf.mul(tf.expand_dims(q_reduct, 1), im_reduct)
        vq_embed = slim.dropout(vq_embed, keep_prob)

        # compute gating logits
        gate_logits = slim.fully_connected(vq_embed, 1, activation_fn=None, scope='gates')

        # compute gating noise
        noise_std = slim.fully_connected(vq_embed, 1, activation_fn=tf.nn.softplus,
                                         scope='noise_std')
        # sample noise
        if phase == 'train':
            batch_size = tf.shape(noise_std)[0]
            _, num_chn, _ = _get_tensor_shape(noise_std)
            noise = tf.random_normal([batch_size, num_chn, 1], dtype=tf.float32)
            noise = tf.reshape(noise, [batch_size, num_chn, 1])
            noise_add = tf.mul(noise_std, noise)
        else:
            noise_add = tf.zeros_like(noise_std, dtype=tf.float32)

        # gate response
        gate_logits = tf.squeeze(gate_logits + noise_add, squeeze_dims=[2])
        gate_logits = tf.exp(gate_logits)

        # compute top k activations
        max_val, _ = tf.nn.top_k(gate_logits, num_active, False)
        mask = tf.gradients(max_val, gate_logits)[0]
        mask = tf.stop_gradient(mask)

        # apply mask
        gate_logits = gate_logits * mask

        # compute softmax
        den = tf.expand_dims(tf.reduce_sum(gate_logits, reduction_indices=1), 1)
        gates = tf.div(gate_logits, den)
        add_vector_summaries('gates', gates)
        add_count_summaries('act_counts', mask)

        # add gate regularizer
        # mean_gates = tf.reduce_mean(gates, reduction_indices=0)  # mean gates response
        # loss = tf.mul(decay_factor, tf.reduce_mean(tf.nn.square(mean_gates)), name='g_reg')
        coeff_var2 = compute_squared_cv(tf.reduce_mean(gates, reduction_indices=0))
        loss = tf.mul(decay_factor, coeff_var2, name='g_reg')
        slim.losses.add_loss(loss)
        tf.scalar_summary('gate_reg_loss', loss)
        add_gradient_summaries('cv_gate_grad', loss, gates)

        # visual pooling
        gates = tf.expand_dims(gates, 2)
        att_pool = tf.reduce_sum(tf.mul(gates, att_banks), reduction_indices=1)
    return att_pool, gates


def seq_moe_attention(im, quest, embed_dim, keep_prob=1.0,
                      num_units=4, reuse=False, scope="SeqMoeAttention"):
    with tf.variable_scope(scope, reuse=reuse):
        im_ctx = mlb(im, quest, embed_dim, keep_prob)
        # soft attention
        s_logits = slim.conv2d(im_ctx, num_units, [1, 1], activation_fn=None,
                               scope='sp_logits')
        # spatial softmax
        att_maps = _spatial_softmax(s_logits)
        # aggregate visual features
        att_basis = _spatial_aggregate(im, att_maps, normalize=False)
        # compute gate
        with tf.variable_scope('Gate'):
            im_ctx = tf.reduce_mean(im, reduction_indices=[1, 2])
            vq_embed = mlb(im_ctx, quest, embed_dim / 4, keep_prob, 'gate_prelogit')
            gates = slim.fully_connected(vq_embed, num_units, activation_fn=None,
                                         scope='gate')
            gates = tf.expand_dims(slim.softmax(gates), 2)
        gated_v = tf.reduce_sum(tf.mul(att_basis, gates), reduction_indices=1)
        # attention flow
        with tf.variable_scope('Flow'):
            gru_cells = tf.nn.rnn_cell.GRUCell(embed_dim, activation=tf.nn.relu)
            gru_cells = tf.nn.rnn_cell.DropoutWrapper(gru_cells,
                                                      output_keep_prob=keep_prob)
            _, att_flow_v = tf.nn.dynamic_rnn(gru_cells, att_basis, dtype=tf.float32, scope='GRnn')
        output = tf.concat(1, [gated_v, att_flow_v])
        return output


def _3d_mlb(att_basis, quest, embed_dim, keep_prob):
    v = slim.fully_connected(att_basis, embed_dim, activation_fn=tf.nn.tanh, scope='v')
    q = slim.fully_connected(quest, embed_dim, activation_fn=tf.nn.tanh, scope='q')
    q = tf.expand_dims(q, 1)
    vq = tf.mul(v, q)
    return slim.dropout(vq, keep_prob=keep_prob)


def qmoe_attention(im, quest, embed_dim, keep_prob=1.0,
                   num_units=4, reuse=False, scope="MoeAttention"):
    with tf.variable_scope(scope, reuse=reuse):
        im_ctx = mlb(im, quest, embed_dim, keep_prob)
        # soft attention
        s_logits = slim.conv2d(im_ctx, num_units, [1, 1], activation_fn=None,
                               scope='sp_logits')
        # spatial softmax
        att_maps = _spatial_softmax(s_logits)
        # aggregate visual features
        att_basis = _spatial_aggregate(im, att_maps, normalize=False)
        # compute gate
        with tf.variable_scope('Gate'):
            gate_basis = _3d_mlb(att_basis, quest, embed_dim, keep_prob)
            gate_logits = slim.fully_connected(gate_basis, 1, activation_fn=None, scope='logits')
            gates = tf.squeeze(gate_logits, squeeze_dims=[2])
            gates = tf.expand_dims(slim.softmax(gates), 2)
        output = tf.reduce_sum(tf.mul(att_basis, gates), reduction_indices=1)
        return output


def debug_moe_attention():
    import numpy.random as nr
    im_arr = nr.rand(2, 14, 14, 2048)
    quest_arr = nr.rand(2, 1024)
    im = tf.convert_to_tensor(im_arr, dtype=tf.float32)
    quest = tf.convert_to_tensor(quest_arr, dtype=tf.float32)
    coding = moe_attention(im, quest, 512, num_units=8)
    sess.run(tf.initialize_all_variables())
    print(coding.eval())


def soft_attention(im, ctx, embed_dim, n_glimpses=1, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        att_feats = []
        for i in range(n_glimpses):
            with tf.variable_scope('glimpse%d' % i):
                att_feats.append(_soft_attention_pool(im, im_ctx)[0])
    return att_feats


def _build_softmax_t(im, ctx, embed_dim, keep_prob, scale=5.0):
    g_im = tf.reduce_mean(im, axis=[1, 2])
    g_embed = mlb(g_im, ctx, embed_dim=embed_dim,
                  keep_prob=keep_prob, scope='g_embed')
    t = slim.fully_connected(g_embed, 1, activation_fn=tf.nn.sigmoid,
                             scope='temp')
    return t * scale


def dynamic_soft_attention(im, ctx, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        with tf.variable_scope('mask'):
            im_ctx = mlb(im, ctx, embed_dim, keep_prob)
            att_logits = slim.conv2d(im_ctx, 1, [1, 1],
                                     activation_fn=None,
                                     scope='att_logits')
            att_logits = tf.reshape(att_logits, [-1, 14*14])
        with tf.variable_scope('temperature'):
            t = _build_softmax_t(im, ctx, embed_dim, keep_prob)
        from gumbel_softmax import gumbel_softmax
        mask = gumbel_softmax(att_logits, t, False)  # soft
        mask = tf.reshape(mask, [-1, 14, 14, 1])
        attend = tf.reduce_sum(im * mask, reduction_indices=[1, 2])
    return attend


def sigmoid_attention(im, ctx, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        with tf.variable_scope('glimpse0'):
            att_feats, att_m = _sigmoid_attention_pool(im, im_ctx)
    return att_feats, att_m


def softmax_attention(im, ctx, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        with tf.variable_scope('glimpse0'):
            att_feats, att_map = _soft_attention_pool(im, im_ctx)
    return att_feats, att_map


def david_attention(im, ctx, n_glimpses=1, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = david_pool(im, ctx)
        att_feats = []
        for i in range(n_glimpses):
            with tf.variable_scope('glimpse%d' % i):
                att_feats.append(_soft_attention_pool(im, im_ctx))
    return att_feats


def attention_cell_helper(im, ctx, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "Att"
    _, h, w, c = im.get_shape().as_list()
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        with tf.variable_scope('glimpse0'):
            v, am = _soft_attention_pool_with_map(im, im_ctx)
        am = tf.reshape(am, shape=[-1, h * w])
    return v, am


def conditional_attention_cell_helper(im, a, part_q, embed_dim, keep_prob=1.0, scope=""):
    scope = scope or "ConditionalAttentionCell"
    _, h, w, c = im.get_shape().as_list()
    with tf.variable_scope(scope):
        # QA joint embedding
        ctx = concat_fusion(part_q, a, embed_dim)
        # soft attention
        im_ctx = mlb(im, ctx, embed_dim, keep_prob, scope='Matching')
        v, am = _soft_attention_pool_with_map(im, im_ctx)
        am = tf.reshape(am, shape=[-1, h * w])
    return v, ctx, am


def basic_attention_helper(im, ctx, embed_dim, keep_prob=1.0, op='concat'):
    assert (op in ['concat', 'mul'])
    if op == 'mul':
        return mul_attention(im, ctx, embed_dim, keep_prob)
    else:
        return concat_attention(im, ctx, embed_dim, keep_prob)


def mul_attention(im, ctx, embed_dim, keep_prob=1.0):
    _, h, w, c = im.get_shape().as_list()
    im_ctx = mlb(im, ctx, embed_dim, keep_prob)
    v, am = _soft_attention_pool_with_map(im, im_ctx)
    am = tf.reshape(am, shape=[-1, h * w])
    return v, am


def concat_attention(im, ctx, embed_dim, keep_prob=1.0):
    _, h, w, c = im.get_shape().as_list()
    im_ctx = mm_conv_concat(im, ctx, embed_dim, keep_prob)
    v, am = _soft_attention_pool_with_map(im, im_ctx)
    am = tf.reshape(am, shape=[-1, h * w])
    return v, am


def soft_attention_with_map(im, ctx, embed_dim,
                            keep_prob=1.0, scope=""):
    scope = scope or "Att"
    with tf.variable_scope(scope):
        im_ctx = mlb(im, ctx, embed_dim, keep_prob)
        avg_v, att = _soft_attention_pool_with_map(im, im_ctx)
    # att = tf.reshape(att, [-1, 14 * 14])
    return avg_v, att


def l_inf_normalize(x, axis=1, EPS=1e-14):
    s = tf.reduce_max(x, reduction_indices=axis)
    l_inf_norm = tf.expand_dims(s, axis) + EPS
    tf.stop_gradient(l_inf_norm)
    return tf.div(x, l_inf_norm, name='l_inf')


def compact_bilinear_pooling(x, y, output_dim):
    x_ = expand_x_y_dims(x)
    y_ = expand_x_y_dims(y)
    z = compt_bilinear_pooling_layer(x_, y_, output_dim)
    return tf.reshape(z, [-1, output_dim])


def slice_array_by_column(t, where):
    batch_size = tf.shape(t)[0]
    n = t.get_shape().as_list()[-1]
    t1 = tf.slice(t, begin=tf.pack([0, 0]), size=tf.pack([batch_size, where]))
    t2 = tf.slice(t, begin=tf.pack([0, where]),
                  size=tf.pack([batch_size, n - where]))
    t1 = tf.reshape(t1, shape=[batch_size, where])
    t2 = tf.reshape(t2, shape=[batch_size, n - where])
    return t1, t2


def slice_3d_tensor_lastd(t, where):
    batch_size = tf.shape(t)[0]
    n_steps = tf.shape(t)[1]
    n = t.get_shape().as_list()[-1]
    t1 = tf.slice(t, begin=tf.pack([0, 0, 0]), size=tf.pack([batch_size, n_steps, where]))
    t1 = tf.reshape(t1, shape=tf.pack([batch_size, n_steps, where]))
    t2 = tf.slice(t, begin=tf.pack([0, 0, where]),
                  size=tf.pack([batch_size, n_steps, n - where]))
    t2 = tf.reshape(t2, shape=tf.pack([batch_size, n_steps, n - where]))
    return t1, t2


def concat_op(values, axis, name='concat'):
    if tf.__version__ == '0.12.0':
        return tf.concat(axis, values, name)
    else:
        return tf.concat(values, axis, name)


def split_op(values, num_splits, axis):
    if tf.__version__ == '0.12.0':
        return tf.split(axis, num_splits, values)
    else:
        return tf.split(values, num_splits, axis)


def select_op(conditon, x, y):
    if tf.__version__ == '0.12.0':
        return tf.select(conditon, x, y)
    else:
        return tf.where(conditon, x, y)


def unpack_op(value, num=None, axis=0, name='unstack'):
    if tf.__version__ == '0.12.0':
        return tf.unpack(value, num, axis, name)
    else:
        return tf.unstack(value, num, axis, name)


def expand_and_tile(t, axis, num):
    """
    Add a new axis and tile n times
    """
    t = tf.expand_dims(t, axis)
    rank = len(t.get_shape().as_list())
    n_tiles = [1] * rank
    n_tiles[axis] = num
    return tf.tile(t, n_tiles)


def replicate_batch(t, num):
    """
    Replicate the sample num times
    """
    try:  # can infer shape
        new_shape = [-1] + t.get_shape().as_list()[1:]
        tt = expand_and_tile(t, 1, num)
        return tf.reshape(tt, new_shape)
    except Exception, e:  # can't infer shape
        n_rank = len(t.get_shape().as_list())
        new_shape = [tf.shape(t)[0] * num] + [tf.shape(t)[i] for i in range(1, n_rank)]
        tt = expand_and_tile(t, 1, num)
        return tf.reshape(tt, new_shape)


def batch_gather_2d(params, indices):
    batch_size = tf.shape(indices)[0]
    datum_len = tf.shape(indices)[1]
    indices = tf.reshape(indices, [-1])
    params = tf.reshape(params, [-1])
    ind = tf.range(batch_size) * datum_len + indices
    return tf.gather(params, ind)


if __name__ == '__main__':
    import numpy.random as nr

    x = nr.rand(4, 4, 5)
    sess = tf.InteractiveSession()
    # t1, t2 = slice_3d_tensor_lastd(tf.constant(x), 2)
    x = slim.fully_connected(tf.constant(x, dtype=tf.float32), 3,
                             activation_fn=None, scope='mem/fc')
    y = prelu(x)
    tf.initialize_all_variables().run()
    print(y.eval())
    print('x:')
    print(x)
    print('\n t1:')

    # debug_moe_attention()
    # x = tf.constant(np.array([[1, 2, 3], [2, 3, 4]]), dtype=tf.float32)
    # sess = tf.Session()
    # print(sess.run(l_inf_normalize(x)))
    # print(sess.run(l_inf_normalize(x, axis=0)))
    # x = tf.placeholder(dtype=tf.float32, shape=[None, 14, 14, 2048])
    # q = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    # z, v = soft_attention(x, q, 1200, 2, 'att')
    # import numpy.random as nr
    #
    # im = nr.rand(10, 14, 14, 2048)
    # quest = nr.rand(10, 2048)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    # atts = sess.run([z, v], feed_dict={x: im, q: quest})
    # print(atts[0])
    # print(atts[0].shape)
    # print(atts[1])
    # print(atts[1].shape)
    # for var in tf.trainable_variables():
    #     print(var.name)
    #     print(var.get_shape())
