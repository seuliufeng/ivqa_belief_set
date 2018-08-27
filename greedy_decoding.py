import tensorflow as tf

_MAX_DEC_LEN = 20


def create_greedy_decoder(init_state, cells, word_map, softmax_params, start_token_id):
    # build dummy decoder inputs
    batch_size = tf.shape(init_state[-1])[0]
    start_tokens = tf.ones(shape=[batch_size], dtype=tf.int32) * start_token_id
    dummy_inputs = [start_tokens] * _MAX_DEC_LEN
    scores, path = embedding_rnn_decoder(decoder_inputs=dummy_inputs,
                                         initial_state=init_state,
                                         cell=cells,
                                         word_embedding=word_map,
                                         output_projection=softmax_params,
                                         feed_previous=True,
                                         scope='RNN')
    scores = _convert_to_tensor(scores)
    path = _convert_to_tensor(path)
    return scores, path


def _convert_to_tensor(inputs):
    from ops import concat_op
    return concat_op([tf.expand_dims(i, 1) for i in inputs], axis=1)


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          word_embedding,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
    loop_function = _extract_argmax_and_embed(
        word_embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (
        tf.nn.embedding_lookup(word_embedding, i) for i in decoder_inputs)
    return rnn_decoder(emb_inp, initial_state, cell,
                       loop_function=loop_function,
                       scope=scope)


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        prev_score = tf.reduce_max(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev, prev_score, prev_symbol

    return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
    with tf.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        scores = []
        path = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp, score, sym = loop_function(prev, i)
                    scores.append(score)
                    path.append(sym)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return scores, path
