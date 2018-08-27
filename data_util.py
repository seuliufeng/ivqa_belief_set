import tensorflow as tf


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
    """Batches input images and captions.

    This function splits the caption into an input sequence and a target sequence,
    where the target sequence is the input sequence right-shifted by 1. Input and
    target sequences are batched and padded up to the maximum length of sequences
    in the batch. A mask is created to distinguish real words from padding words.

    Example:
      Actual captions in the batch ('-' denotes padded character):
        [
          [ 1 2 5 4 5 ],
          [ 1 2 3 4 - ],
          [ 1 2 3 - - ],
        ]

      input_seqs:
        [
          [ 1 2 3 4 ],
          [ 1 2 3 - ],
          [ 1 2 - - ],
        ]

      target_seqs:
        [
          [ 2 3 4 5 ],
          [ 2 3 4 - ],
          [ 2 3 - - ],
        ]

      mask:
        [
          [ 1 1 1 1 ],
          [ 1 1 1 0 ],
          [ 1 1 0 0 ],
        ]

    Args:
      images_and_captions: A list of pairs [image, caption], where image is a
        Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
        any length. Each pair will be processed and added to the queue in a
        separate thread.
      batch_size: Batch size.
      queue_capacity: Queue capacity.
      add_summaries: If true, add caption length summaries.

    Returns:
      images: A Tensor of shape [batch_size, height, width, channels].
      input_seqs: An int32 Tensor of shape [batch_size, padded_length].
      target_seqs: An int32 Tensor of shape [batch_size, padded_length].
      mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
    """
    enqueue_list = []
    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.sub(caption_length, 1), 0)

        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, target_seq, indicator])

    images, input_seqs, target_seqs, mask = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad")


    return images, input_seqs, target_seqs, mask
