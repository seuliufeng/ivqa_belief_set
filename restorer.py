import tensorflow as tf
import os


class VarRenameHelper(object):
    def __init__(self):
        rnn_var_mapping = {
            '/BasicLSTMCell/Linear/Matrix': '/basic_lstm_cell/weights',
            '/BasicLSTMCell/Linear/Bias': '/basic_lstm_cell/biases',
            '/LSTMCell/W_0': '/lstm_cell/weights',
            '/LSTMCell/B': '/lstm_cell/biases'
        }
        reverse_mappling = {v: k for (k, v) in rnn_var_mapping.iteritems()}
        rnn_var_mapping.update(reverse_mappling)
        self.rnn_var_mapping = rnn_var_mapping

    def check_var(self, name):
        for k in self.rnn_var_mapping.keys():
            if k in name:
                name = name.replace(k, self.rnn_var_mapping[k])
                break
        return name


def create_variable_list(convert=False, warpper_scope=None, model_vars=None):
    if model_vars is None:
        model_vars = tf.global_variables()

    if type(model_vars) == list:
        model_vars = {var.name.split(':')[0]: var for var in model_vars}

    if not convert:
        if warpper_scope is None:
            return model_vars

    # need to convert version
    var_list = {}
    helper = VarRenameHelper()
    for var in model_vars.values():
        if convert:
            name_in_ckpt = helper.check_var(var.name)
        else:
            name_in_ckpt = var.name
        name_in_ckpt = name_in_ckpt.split(':')[0]
        if warpper_scope:
            name_in_ckpt = os.path.join(warpper_scope, name_in_ckpt)
        var_list[name_in_ckpt] = var
    return var_list


class Restorer(object):
    def __init__(self, graph, warpper_scope=None):
        with graph.as_default():
            default_var = create_variable_list(convert=False, warpper_scope=warpper_scope)
            self._saver_default = tf.train.Saver(default_var)
            convert_var = create_variable_list(convert=True, warpper_scope=warpper_scope)
            self._saver_convert = tf.train.Saver(convert_var)

    def restore(self, sess, checkpoint_path):
        tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
        try:
            self._saver_default.restore(sess, checkpoint_path)
        except Exception as e:
            tf.logging.info('Using converted version')
            self._saver_convert.restore(sess, checkpoint_path)


def create_restore_fn_remove_scope(scope_name, ckpt_path, exclude_scope=None,
                                   convert=False):
    # create restore function
    if scope_name is None:
        vqa_vars = tf.global_variables()
    else:
        vqa_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name)
    if exclude_scope:
        vqa_vars = [var for var in vqa_vars if exclude_scope not in var.name]

    tmp_var_dict = create_variable_list(convert=convert, model_vars=vqa_vars)

    def rename_fn(vname):
        return '/'.join(vname.split('/')[1:])

    var_dict = {rename_fn(var_name): var for var_name, var in tmp_var_dict.iteritems()}
    saver = tf.train.Saver(var_list=var_dict)
    print('%s: Added %d variables to init op' % (scope_name or 'Vars', len(var_dict.keys())))

    def restore_fn(sess):
        tf.logging.info("Restoring weights of VQA agent from checkpoint file %s",
                        ckpt_path)
        saver.restore(sess, ckpt_path)

    return restore_fn


def create_restore_fn(ckpt_path, exclude_scope=None,
                      convert=False,
                      replace_rule=None):
    # create restore function
    vqa_vars = tf.trainable_variables()

    if exclude_scope:
        vqa_vars = [var for var in vqa_vars if exclude_scope not in var.name]

    tmp_var_dict = create_variable_list(convert=convert, model_vars=vqa_vars)
    # import pdb
    # pdb.set_trace()
    var_dict = {}
    for var_name, var in tmp_var_dict.iteritems():
        if replace_rule is not None:
            for rule in replace_rule:
                var_name = var_name.replace(rule[0], rule[1])
        var_dict[var_name] = var

    # var_dict = {var_name: var for var_name, var in tmp_var_dict.iteritems()}

    saver = tf.train.Saver(var_list=var_dict)

    # print('==================================')
    # print('%s: Added %d variables to init op' % (scope_name or 'Vars', len(var_dict.keys())))
    # for var in var_dict.values():
    #     print('%s:' % var.name)
    #     print(var.get_shape())
    # import pdb
    # pdb.set_trace()

    def restore_fn(sess):
        tf.logging.info("Restoring weights of VQA agent from checkpoint file %s",
                        ckpt_path)
        saver.restore(sess, ckpt_path)

    return restore_fn


def merge_init_fns(init_fns):
    def restore_fn(sess):
        for init_fn in init_fns:
            init_fn(sess)

    return restore_fn


if __name__ == '__main__':
    pass
