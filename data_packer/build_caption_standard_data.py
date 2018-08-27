import re
import numpy as np
import json
import os
import tensorflow as tf
import nltk.tokenize
import sys
from datetime import datetime
from collections import OrderedDict, namedtuple, Counter
from util import save_hdf5, save_json, load_json

############################################################
################### CAPTION HELPERS ########################
############################################################
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("output_dir", "data/", "Output data directory.")
FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "captions"])


def _process_caption(caption):
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def _load_and_process_metadata(captions_file):
    with tf.gfile.FastGFile(captions_file, "r") as f:
        caption_data = json.load(f)

    # Extract the filenames.
    id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

    # Extract the captions. Each image_id is associated with multiple captions.
    id_to_captions = {}
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)

    assert len(id_to_filename) == len(id_to_captions)
    assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
    print("Loaded caption metadata for %d images from %s" %
          (len(id_to_filename), captions_file))

    # Process the captions and combine the data into a list of ImageMetadata.
    print("Proccessing captions.")
    image_metadata = []
    num_captions = 0
    for image_id, base_filename in id_to_filename:
        captions = [_process_caption(c) for c in id_to_captions[image_id]]
        image_metadata.append(ImageMetadata(image_id, captions))
        num_captions += len(captions)
    print("Finished processing %d captions for %d images in %s" %
          (num_captions, len(id_to_filename), captions_file))
    return image_metadata


def _create_vocab(captions):
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))
    return word_counts


############################################################
###################      UTILS     #########################
############################################################

def _dump_dictonary(vocab, fname):
    word_counts = vocab.items()
    with tf.gfile.FastGFile(fname, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file: %s" % fname)


class Vocabulary(object):
    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    @property
    def unk_id(self):
        return self._unk_id


class CaptionEncoder(object):
    def __init__(self, vocab, max_len=30):
        self._vocab = vocab
        self._max_len = max_len
        self.vocab_size = vocab.unk_id

    def encode(self, caption):
        seq = [self._vocab.word_to_id(word) for word in caption]
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]
        return seq


############################################################
###################   VQA HELPERS   ########################
############################################################

def find_image_id_from_fname(filename):
    return int(re.findall('\d+', filename)[-1])


def get_subset_image_id(subset):
    print('Parsing image index of set %s...' % subset)
    meta_file = 'data/vqa_std_mscoco_%s.meta' % subset
    images = load_json(meta_file)['images']
    # get image ids
    image_ids = [find_image_id_from_fname(fname) for fname in images]
    unique_ids = np.unique(np.array(image_ids, dtype=np.int32))
    # print('Finshed, find %d unique images.\n' % unique_ids.size)
    return unique_ids.tolist()


def load_vqa_vocab():
    print('Loading VQA vocabularies...')
    start_word = '<S>'
    end_word = '</S>'
    vocab_file = 'data/vqa_trainval_question_word_counts.txt'
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    items = [(line.split()[0], int(line.split()[1])) for line in reverse_vocab]
    vocab = OrderedDict(items)
    assert (start_word in vocab and end_word in vocab)
    print('Finished, loaded %d words from VQA question vocab.\n' % len(vocab))
    return vocab


def merge_vocabs(vqa_vocab, caption_vocab):
    print('Updating VQA vocabs with caption vocabs')
    vqa_vocab_size = len(vqa_vocab)
    for word, count in caption_vocab:
        if word in vqa_vocab:
            vqa_vocab[word] += count
        else:  # create new entry
            vqa_vocab[word] = count
    new_vocab_size = len(vqa_vocab)
    _dump_dictonary(vqa_vocab, 'data/MTL_trainval_word_counts.txt')
    print('vocab size before merging: %d' % vqa_vocab_size)
    print('vocab size after merging: %d\n' % new_vocab_size)
    # build vocabulary object
    vocab = OrderedDict([(word, word_id) for word_id, word in enumerate(vqa_vocab)])
    unk_id = len(vocab)
    vocab = Vocabulary(vocab, unk_id)
    return vocab


def load_caption_meta(subset):
    annotation_dir = '/import/vision-ephemeral/fl302/mscoco/raw_data/annotations'
    caption_file = os.path.join(annotation_dir, 'captions_%s2014.json' % subset)
    return _load_and_process_metadata(caption_file)


def split_datasets(meta, split):
    print('\nSpliting subset <%s>' % split)
    image_inds = get_subset_image_id(split)
    image_inds = {id: 0 for id in image_inds}  # build hashing table for query
    target = [info for info in meta if info.image_id in image_inds]
    assert(len(target) == len(image_inds))
    print('Finished, %s samples are collected\n' % len(target))
    return target


def _process_dataset(subset, images, encoder):
    data_filename = os.path.join(FLAGS.output_dir, 'caption_std_mscoco_%s.data' % subset)

    num_images = len(images)
    capts = []
    image_ids = []
    for i, info in enumerate(images):
        image_id = info.image_id
        for c in info.captions:
            capts.append(encoder.encode(c))
            image_ids.append(image_id)

        if not i % 1000:
            print("%s: Processed %d of %d items." % (datetime.now(), i, num_images))
            sys.stdout.flush()

    # merge questions to a matrix
    seq_len = [len(q) for q in capts]
    max_len = max(seq_len)
    num_capts = len(capts)
    dummy_id = encoder.vocab_size + 1
    capt_arr = np.ones([num_capts, max_len], dtype=np.int32)*dummy_id
    for i, x in enumerate(capt_arr):
        x[:seq_len[i]] = capts[i]
    seq_len = np.array(seq_len, dtype=np.int32)
    image_ids = np.array(image_ids, dtype=np.int32)
    # save data file
    save_hdf5(data_filename, {'capt_arr': capt_arr,
                              'capt_len': seq_len,
                              'image_ids': image_ids})

    print("%s: Wrote %d caption files to %s" %
          (datetime.now(), num_images, data_filename))
    sys.stdout.flush()


if __name__ == '__main__':
    # fetch captioning meta data
    meta_train = load_caption_meta('train')
    meta_val = load_caption_meta('val')
    meta = meta_train + meta_val
    # split dataset
    train_set = split_datasets(meta, 'trainval')
    dev_set = split_datasets(meta, 'dev')
    # build vocabulary
    captions = [c for image in train_set for c in image.captions]
    caption_vocab = _create_vocab(captions)
    vqa_vocab = load_vqa_vocab()
    merged_vocab = merge_vocabs(vqa_vocab, caption_vocab)
    # encode the captions
    encoder = CaptionEncoder(merged_vocab)
    _process_dataset('trainval', train_set, encoder)
    _process_dataset('dev', dev_set, encoder)
