import tensorflow as tf

_MAX_DEC_LEN = 20


def create_greedy_decoder(init_state, cells, word_map, softmax_params,
                          start_token_id, epsion=1.0):
    # build dummy decoder inputs
    reuse = False
    batch_size = tf.shape(init_state[-1])[0]
    if type(start_token_id) == int:
        start_tokens = tf.ones(shape=[batch_size], dtype=tf.int32) * start_token_id
        dummy_inputs = [start_tokens] * _MAX_DEC_LEN
    elif type(start_token_id) == tf.Tensor:  # mixer
        dummy_inputs = [start_token_id] * _MAX_DEC_LEN
        reuse = True
    else:
        raise Exception('unknown input type')
    with tf.variable_scope('RNN', reuse=reuse) as sc:
        path, scores = embedding_rnn_decoder(decoder_inputs=dummy_inputs,
                                             initial_state=init_state,
                                             cell=cells,
                                             word_embedding=word_map,
                                             output_projection=softmax_params,
                                             feed_previous=True,
                                             scope=sc,
                                             epsion=epsion)
    path = _convert_to_tensor(path)
    scores = _convert_to_tensor(scores)
    return path, scores


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
                          scope=None,
                          epsion=1.0):
    loop_function = _extract_mixed_and_embed(
        word_embedding, output_projection,
        update_embedding_for_previous, epsion) if feed_previous else None
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
        return emb_prev, prev_symbol, prev_score

    return loop_function


def _gather_by_index(scores, inds):
    scores = tf.nn.log_softmax(scores)
    scores_flat = tf.reshape(scores, [-1])
    batch_size = tf.shape(scores)[0]
    vocab_size = tf.shape(scores)[1]
    _indices = tf.range(batch_size) * vocab_size + tf.cast(inds, tf.int32)
    values = tf.gather(scores_flat, _indices)
    return values


def _rand_draw(logits):
    _rand_symbol = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    return tf.cast(_rand_symbol, tf.int32)


def _mixed_sampling(logits, epsion=1.0):
    batch_size = tf.shape(logits)[0]
    _seed = tf.random_uniform(shape=(batch_size,), dtype=tf.float32)
    _use_rand = tf.cast(_seed < epsion, tf.int32)
    _rand_symbol = _rand_draw(logits)
    # completely random
    # dummy = tf.random_uniform(shape=[batch_size, 15955])
    dummy = logits
    _max_symbol = tf.cast(tf.argmax(dummy, 1), tf.int32)
    symbol = _rand_symbol * _use_rand + _max_symbol * (1 - _use_rand)
    return symbol


def _extract_mixed_and_embed(embedding, output_projection,
                             update_embedding=True, epsion=1.0):
    def loop_function(prev, _):
        # compute logits
        prev = tf.nn.xw_plus_b(
            prev, output_projection[0], output_projection[1])
        prev_symbol = _mixed_sampling(prev, epsion=epsion)
        prev_scores = _gather_by_index(prev, prev_symbol)
        # prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev, prev_symbol, prev_scores

    return loop_function


def _extract_multinomial_and_embed(embedding, output_projection,
                                   update_embedding=True):
    def loop_function(prev, _):
        # compute logits
        prev = tf.nn.xw_plus_b(
            prev, output_projection[0], output_projection[1])
        prev_symbol = tf.squeeze(tf.multinomial(prev, 1), axis=1)
        prev_scores = _gather_by_index(prev, prev_symbol)
        # prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev, prev_symbol, prev_scores

    return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
    with tf.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        path = []
        score = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp, sym, scr = loop_function(prev, i)
                    path.append(sym)
                    score.append(scr)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return path, score
