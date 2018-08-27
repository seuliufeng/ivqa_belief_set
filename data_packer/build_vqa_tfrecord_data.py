from random import shuffle, seed
import sys
import os
import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import json
from collections import namedtuple, Counter
import tensorflow as tf
from time import time
from scipy.io import loadmat
from datetime import datetime
from util import load_hdf5
from word2vec_util import Word2VecEncoder

tf.flags.DEFINE_string("annotation_dir", "data/annotations",
                       "Directory containing all annotation files.")

tf.flags.DEFINE_string("word2vec_model", "data/GoogleNews-vectors-negative300.bin",
                       "Path of pretrained word2vec model.")
tf.flags.DEFINE_string("output_dir", "data/", "Output data directory.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 0,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_integer("num_top_answers", 2000,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "data/%s_word_counts.txt",
                       "Output vocabulary file of word counts.")
tf.flags.DEFINE_string("top_answer_output_file", "data/%s_top%d_answers.txt",
                       "Output vocabulary file of word counts.")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "question_id", "question", "answer", "token_answer", "choices"])


def _tokenize_sentence(sentence):
    tokenized = [FLAGS.start_word]
    tokenized.extend(word_tokenize(str(sentence).lower()))
    tokenized.append(FLAGS.end_word)
    return tokenized


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


class VQADataEncoder(object):
    def __init__(self):
        self._encoders = []

    def register_encoder(self, enc):
        self._encoders.append(enc)
        self.print_encoder_info(enc)

    def encode(self, image):
        context_dic = {}
        feature_dic = {}

        for enc in self._encoders:
            try:
                if enc.type == 'context':
                    context_dic[enc.target] = enc.encode(image)
                elif enc.type == 'feature':
                    feature_dic[enc.target] = enc.encode(image)
                else:
                    raise Exception('unknown target')
            except KeyboardInterrupt:
                raise
            except Exception, e:
                print('[%s] error in processing image %s, skipping' % (enc.name, image.filename))
                return
        context = tf.train.Features(feature=context_dic)
        feature_lists = tf.train.FeatureLists(feature_list=feature_dic)
        return tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

    def finalize(self):
        print('\n')
        for enc in self._encoders:
            self.print_encoder_info(enc)

    @staticmethod
    def print_encoder_info(enc):
        print('%s encoder:' % enc.name)
        print('type: %s' % enc.type)
        print('target: %s' % enc.target)
        print('field: %s\n' % enc.field)


class Encoder(object):
    def __init__(self):
        self._type = ''
        self._name = ''
        self._target = ''
        self._field = ''

    def encode(self, image):
        pass

    @property
    def target(self):
        return self._target

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def field(self):
        return self._field


class QuestionIdEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'QuestionIdEncoder'
        self._type = 'context'
        self._target = 'image/question_id'
        self._field = 'question_id'

    def encode(self, image):
        quest_id = image.question_id
        return _int64_feature(quest_id)


class ImageIdEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ImageIdEncoder'
        self._type = 'context'
        self._target = 'image/image_id'
        self._field = 'image_id'

    def encode(self, image):
        image_id = image.image_id
        return _int64_feature(image_id)


class ImageVGG19Encoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ImageVGG19Encoder'
        self._type = 'context'
        self._target = 'image/image_vgg19'
        self._field = 'filename'
        images, self._feats = _load_image_encoder()
        self._im2idx = {im: idx for idx, im in enumerate(images)}

    def get_feature(self, filename):
        return self._feats[self._im2idx[filename]]

    def encode(self, image):
        feat = self.get_feature(image.filename)
        return _bytes_feature(feat.tobytes())


class ResNet152Encoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ResNet152Encoder'
        self._type = 'context'
        self._target = 'image/image_resnet152'
        self._field = 'image_id'
        images, self._feats = self._load_image_features()
        self._im2idx = {im: idx for idx, im in enumerate(images)}

    def _load_image_features(self):
        SPLITS = ['val_full', 'train', 'test-dev_full']
        data_root = '/import/vision-ephemeral/fl302/code/text-to-image'

        def _load_subset(subset):
            fname = 'mscoco_res152_%s.h5' % subset
            print('Loading file %s' % fname)
            d = load_hdf5(os.path.join(data_root, fname))
            ids = d['image_ids'].tolist()
            f = d['features']
            return ids, f

        image_ids = []
        feats = []
        for split in SPLITS:
            ids, feat = _load_subset(split)
            image_ids += ids
            feats.append(feat)
        return image_ids, np.concatenate(feats)

    def get_feature(self, image_id):
        return self._feats[self._im2idx[image_id]]

    def encode(self, image):
        feat = self.get_feature(image.image_id)
        return _bytes_feature(feat.tobytes())


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()
        self._resize_dim = 448

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decoded_image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)
        self._re_encoded_jpg = self.create_resize_graph()

    def create_resize_graph(self):
        resized = tf.image.resize_images(self._decoded_image,
                                         size=[self._resize_dim, self._resize_dim],
                                         method=tf.image.ResizeMethod.BILINEAR)
        return tf.image.encode_jpeg(tf.cast(resized, tf.uint8))

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def resize_and_encode(self, image):
        return self._sess.run(self._re_encoded_jpg,
                              feed_dict={self._decoded_image: image})


class ImageEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ImageEncoder'
        self._type = 'context'
        self._target = 'image/encoded'
        self._field = 'filename'
        self._im_root = '/import/vision-ephemeral/fl302/data/VQA/Images/mscoco'
        self._decoder = ImageDecoder()

    def encode(self, image):
        im_file = os.path.join(self._im_root, image.filename)
        with tf.gfile.FastGFile(im_file, "r") as f:
            encoded_image = f.read()
        tmp = self._decoder.decode_jpeg(encoded_image)
        encoded_image = self._decoder.resize_and_encode(tmp)
        return _bytes_feature(encoded_image)


class ResNet152AttEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ResNet152AttEncoder'
        self._type = 'context'
        self._target = 'image/image_resnet152att'
        self._field = 'file_name'

    def get_feature(self, filename):
        feat_root = '/import/vision-ephemeral/fl302/data/VQA/ResNet152/resnet_res5c'
        feat_file = os.path.join(feat_root, filename+'.npz')
        return np.load(feat_file)['x']

    def encode(self, image):
        feat = self.get_feature(image.filename)
        return _bytes_feature(feat.tobytes())


class ImageInceptionProbEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)
        self._name = 'ImageInceptionProbEncoder'
        self._type = 'context'
        self._target = 'image/image_incept_prob'
        self._field = 'image_id'
        self._prob_path = 'data/Inception_v3_1000_Trainval_Dets.mat'
        d = loadmat(self._prob_path)
        self._probs = d['probs'].astype(np.float32)
        index = d['index'].astype(np.int32).flatten()
        self._image_id_to_index = dict([(image_ids, i) for (i, image_ids) in enumerate(index)])

    def get_feature(self, image_id):
        idx = self._image_id_to_index[image_id]
        return self._probs[idx, :]

    def encode(self, image):
        feat = self.get_feature(image.image_id)
        return _bytes_feature(feat.tobytes())


class QuestionEncoder(Encoder):
    def __init__(self, vocab_file):
        Encoder.__init__(self)
        self._name = 'QuestionEncoder'
        self._type = 'feature'
        self._target = 'image/question_ids'
        self._field = 'question'
        self._vocab = _load_vocab(vocab_file)

    def encode(self, image):
        ids = [self._vocab.word_to_id(word) for word in image.question]
        return _int64_feature_list(ids)


class AnswerEncoder(Encoder):
    def __init__(self, coding, vocab_file=None):
        Encoder.__init__(self)
        self._coding = coding
        self._name = 'AnswerEncoder-%s' % (coding.title())
        self._type = 'feature' if coding == 'sequence' else 'context'
        self._target = 'image/answer_%s' % coding

        if self._coding == 'sequence':
            self._field = 'token_answer'
            self._vocab = _load_vocab(vocab_file)
        elif self._coding == 'one_hot':
            self._field = 'answer'
            self._vocab = _load_answer_vocab(vocab_file)
        elif self._coding == 'word2vec':
            self._field = 'token_answer'
            self._vocab = Word2VecEncoder('data/vqa_word2vec_model.pkl')

    def encode_sequence(self, ans):
        ans = ['\S'] if ans is None else ans
        ids = [self._vocab.word_to_id(word) for word in ans]
        return _int64_feature_list(ids)

    def encode_word_vector(self, ans):
        ans = ['\S'] if ans is None else ans
        feat = self._vocab.encode(ans)
        return _bytes_feature(feat.tobytes())

    def encode_one_hot(self, ans):
        idx = 0 if ans is None else self._vocab.word_to_id(ans)
        return _int64_feature(idx)

    def encode(self, image):
        if self._coding == 'sequence':
            ans = image.token_answer
            return self.encode_sequence(ans)
        elif self._coding == 'one_hot':
            ans = image.answer
            return self.encode_one_hot(ans)
        elif self._coding == 'word2vec':
            ans = image.token_answer
            return self.encode_word_vector(ans)
        else:
            raise Exception('unknown coding')


def create_sample_encoder(trainset='trainval'):
    encoder = VQADataEncoder()
    # image id encoder
    encoder.register_encoder(ImageIdEncoder())
    # image feature encoder
    # encoder.register_encoder(ImageVGG19Encoder())
    # semantic feature encoder
    # encoder.register_encoder(ResNet152Encoder())
    # encoder.register_encoder(ResNet152AttEncoder())
    encoder.register_encoder(ImageEncoder())
    # question id encoder
    encoder.register_encoder(QuestionIdEncoder())
    # question encoder
    vocab_name = 'vqa_%s_question' % trainset
    quest_vocab_file = FLAGS.word_counts_output_file % vocab_name
    encoder.register_encoder(QuestionEncoder(quest_vocab_file))
    # answer encoder
    vocab_name = 'vqa_%s_answer' % trainset
    ans_vocab_file = FLAGS.word_counts_output_file % vocab_name
    encoder.register_encoder(AnswerEncoder('sequence', ans_vocab_file))
    # word2vec encoder
    word2vec_file = 'data/vqa_word2vec_model.pkl'
    encoder.register_encoder(AnswerEncoder('word2vec', word2vec_file))
    # frequent answer encoder
    top_ans_file = FLAGS.top_answer_output_file % ('vqa_%s' % trainset, FLAGS.num_top_answers)
    encoder.register_encoder(AnswerEncoder('one_hot', top_ans_file))
    encoder.finalize()
    return encoder


def _load_image_encoder():
    meta_file = 'data/full/full_images.json'
    data_file = 'data/full/full_image_feats.h5'
    feats = []
    with h5py.File(data_file, 'r') as hf:
        feats.append(np.array(hf.get('images_train')).astype(np.float32))
        feats.append(np.array(hf.get('images_test')).astype(np.float32))
    feats = np.concatenate(feats, axis=0)
    # apply l2 normalize
    l2norm = np.sqrt(np.sum(np.square(feats), axis=1)).reshape([-1, 1])
    feats = np.divide(feats, l2norm)

    d = _read_json('', meta_file)
    images = d['unique_img_train'] + d['unique_img_test']
    return images, feats


def _create_top_answers_vocab(images, nkeep=1000, subset='train'):
    counts = {}
    for image in images:
        ans = image.answer
        counts[ans] = counts.get(ans, 0) + 1

    ans_count = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'
    print '\n'.join(map(str, ans_count[:20]))

    top_ans = [ans_count[i][1] for i in range(nkeep)]
    output_file = FLAGS.top_answer_output_file % ('vqa_%s' % subset, nkeep)
    with tf.gfile.FastGFile(output_file, "w") as f:
        f.write("\n".join(["%s" % item for item in top_ans]))
    print("Wrote vocabulary file:", output_file)

    ans_vocab = {ans: i for (i, ans) in enumerate(top_ans)}
    return Vocabulary(ans_vocab, len(ans_vocab))


def filter_question(images, ans_vocab):
    unk_id = ans_vocab.unk_id
    flt_imgs = [im for im in images if ans_vocab.word_to_id(im.answer) != unk_id]
    print 'question number reduce from %d to %d ' % (len(images), len(flt_imgs))
    print 'preserved %%%0.2f questions' % (len(flt_imgs) * 100.0 / len(images))
    return flt_imgs


def _read_json(data_root, filename, key=None):
    d = json.load(open(os.path.join(data_root,
                                    filename), 'r'))
    return d if key is None else d[key]


def _load_and_process_metadata(subset):
    tf.logging.info('Processing meta data of %s...' % subset)
    t = time()
    is_test = subset.startswith('test')
    year = 2015 if is_test else 2014
    subtype = '%s%d' % (subset, year)
    ann_root = FLAGS.annotation_dir
    datatype = 'test2015' if is_test else subtype
    IMFORMAT = '%s/COCO_%s_%012d.jpg'
    # tf.logging.info('Loading annotations and questions...')
    questions = _read_json(ann_root, 'MultipleChoice_mscoco_%s_questions.json' % subtype, 'questions')
    dataset = questions if is_test \
        else _read_json(ann_root, 'mscoco_%s_annotations.json' % subtype, 'annotations')

    meta = []
    for info, quest in zip(dataset, questions):
        ans = None if is_test else info['multiple_choice_answer']
        token_ans = None if is_test else _tokenize_sentence(ans)
        quest_id = info['question_id']
        image_id = info['image_id']
        filename = IMFORMAT % (datatype, datatype, info['image_id'])
        question = _tokenize_sentence(quest['question'])
        mc_ans = quest['multiple_choices']
        meta.append(ImageMetadata(image_id, filename, quest_id, question, ans, token_ans, mc_ans))
    tf.logging.info('Time %0.2f sec.' % (time() - t))
    return meta


class Vocabulary(object):
    """Simple vocabulary wrapper."""

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


def _load_vocab(vocab_file):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    reverse_vocab = [line.split()[0] for line in reverse_vocab]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    return Vocabulary(vocab_dict, unk_id)


def _load_answer_vocab(vocab_file):
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        reverse_vocab = list(f.readlines())
    reverse_vocab = [line.strip() for line in reverse_vocab]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    return Vocabulary(vocab_dict, unk_id)


def _create_vocab(captions, vocab_name):
    assert (type(captions[0] == list))
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    word_counts_output_file = FLAGS.word_counts_output_file % vocab_name
    with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    return Vocabulary(vocab_dict, unk_id)


def _create_question_vocab(images, trainset):
    questions = [image.question for image in images]
    return _create_vocab(questions, 'vqa_%s_question' % trainset)


def _create_answer_vocab(images, trainset):
    answers = [image.token_answer for image in images if image.answer is not None]
    assert (type(answers[0]) == list)
    return _create_vocab(answers, 'vqa_%s_answer' % trainset)


def _process_dataset(subset, images, encoder):
    output_filename = 'vqa_jpg_mscoco_%s.tfrecords' % subset
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    # compress = tf.python_io.TFRecordCompressionType.ZLIB
    # option = tf.python_io.TFRecordOptions(compression_type=compress)
    # writer = tf.python_io.TFRecordWriter(output_file, options=option)
    writer = tf.python_io.TFRecordWriter(output_file)

    num_images = len(images)
    for i in range(num_images):
        image = images[i]
        sequence_example = encoder.encode(image)
        if sequence_example is not None:
            writer.write(sequence_example.SerializeToString())

        if not i % 1000:
            print("%s: Processed %d of %d items." % (datetime.now(), i, num_images))
            sys.stdout.flush()

    writer.close()
    print("%s: Wrote %d VQA files to %s" %
          (datetime.now(), num_images, output_file))
    sys.stdout.flush()


def split_dataset(dataset, train_ratio=0.9):
    def _create_index_dict(images):
        print('creating index...')
        d = {}
        for image in images:
            id = image.image_id
            if id in d:
                d[id].append(image)
            else:
                d[id] = [image]
        return d

    print('splitting dataset...')
    id2meta = _create_index_dict(dataset)
    print('splitting...')
    uni_im_ids = id2meta.keys()
    seed(123)
    shuffle(uni_im_ids)
    num_images = len(uni_im_ids)
    num_train = int(num_images * train_ratio)
    train_ids = uni_im_ids[:num_train]
    test_ids = uni_im_ids[num_train:]
    print('unfolding...')
    train = [m for i in train_ids for m in id2meta[i]]
    test = [m for i in test_ids for m in id2meta[i]]
    print('done\n')
    print('============ statistics =============')
    print('Dataset contains %d questions' % len(dataset))
    print('Train set: %d images: Questions: %d' % (len(train_ids), len(train)))
    print('Test set: %d images: Questions: %d' % (len(test_ids), len(test)))
    return train, test


def main(_):
    # from util import wait_to_start
    # im_feat_file = 'data/full/full_images.json'
    # wait_to_start(im_feat_file)  # wait file to start model

    train = _load_and_process_metadata('train')
    val = _load_and_process_metadata('val')
    val1, val2 = split_dataset(dataset=val, train_ratio=0.9)
    train, test = train + val1, val2
    seed(123)
    shuffle(train)
    # build vocabularies
    tf.logging.info('got %d images for training and %d '
                    'images for testing' % (len(train), len(test)))
    tf.logging.info('creating popular answer vocabulary')
    _create_top_answers_vocab(train, FLAGS.num_top_answers, subset='trainval')
    # train = filter_question(train, ans_vocab)
    tf.logging.info('creating question vocabulary')
    _create_question_vocab(train, 'trainval')
    tf.logging.info('creating answer vocabulary')
    _create_answer_vocab(train, 'trainval')
    # setup encoder
    encoder = create_sample_encoder('trainval')
    tf.logging.info('converting development set')
    _process_dataset('dev', test, encoder)
    tf.logging.info('converting trainval')
    _process_dataset('trainval', train, encoder)



# def main(_):
#     # from util import wait_to_start
#     # im_feat_file = 'data/full/full_images.json'
#     # wait_to_start(im_feat_file)  # wait file to start model
#
#     test_dev = _load_and_process_metadata('test-dev')
#     # setup encoder
#     encoder = create_sample_encoder('trainval')
#     tf.logging.info('converting test-dev')
#     _process_dataset('test-dev', test_dev, encoder)


if __name__ == "__main__":
    tf.app.run()
