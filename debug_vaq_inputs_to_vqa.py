import tensorflow as tf
from readers.ivqa_reader_creater import create_reader
from ivqa_inputs_propressing import process_input_data


def _build_vqa_inputs(quest, quest_len):
    pad_token = 15954
    reset_mask = tf.equal(quest, pad_token)
    keep_mask = 1 - tf.cast(reset_mask, tf.int32)
    vqa_quest_in = quest * keep_mask
    vqa_quest_in = vqa_quest_in[:, 1:]  # remove start token
    vqa_quest_len = quest_len - 2  # remove start and end token
    return vqa_quest_in, vqa_quest_len


def test():
    create_fn = create_reader('VAQ-Var', 'test')
    reader = create_fn(batch_size=4)
    reader.start()
    inputs = reader.pop_batch()

    d_quest = inputs[1]
    d_quest_len = inputs[2]
    inputs = process_input_data([None, d_quest, d_quest_len], 15954)
    d_quest = inputs[1]
    d_quest_len = inputs[2]

    def print_array(arr, name):
        print('%s:' % name)
        print(arr)

    quest = tf.placeholder(tf.int32, [None, None])
    quest_len = tf.placeholder(tf.int32, None)
    vqa_quest_in, vqa_quest_len = _build_vqa_inputs(quest, quest_len)

    sess = tf.Session()
    print_array(d_quest, 'q')
    print_array(d_quest_len, 'q_len')

    vqa_q, vqa_q_len = sess.run([vqa_quest_in, vqa_quest_len],
                                feed_dict={quest: d_quest,
                                           quest_len: d_quest_len})
    print_array(vqa_q, 'vqa_q')
    print_array(vqa_q_len, 'vqa_q_len')
    reader.stop()


if __name__ == '__main__':
    test()
