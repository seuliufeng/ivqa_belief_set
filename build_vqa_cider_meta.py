import tensorflow as tf
import cPickle
from collections import defaultdict
from extract_vqa_question_ngram_data import load_and_process_metadata, split_data_by_seed
from nltk.tokenize import word_tokenize

_ADD_END = True
_END_TOKEN = '</S>'


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


# ********************* N gram utils *********************
def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def get_document_stastic(meta, vocab):
    count_imgs = 0
    unk_id = vocab.unk_id

    refs_words = []
    refs_idxs = []
    for img in meta:
        ref_words = []
        ref_idxs = []
        tmp_tokens = word_tokenize(img.question.lower())
        if _ADD_END:
            tmp_tokens += [_END_TOKEN]
        tmp_tokens = [word if vocab.word_to_id(word) != unk_id else 'UNK' for word in tmp_tokens]
        ref_words.append(' '.join(tmp_tokens))
        ref_idxs.append(' '.join([str(vocab.word_to_id(word)) for word in tmp_tokens]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs


def _process_subset(meta, subset):
    images = split_data_by_seed(meta, subset)
    vocab = _load_vocab('data/vqa_trainval_question_word_counts.txt')

    ngram_words, ngram_idxs = get_document_stastic(images, vocab)

    end_suffix = '_end' if _ADD_END else ''
    word_file = 'data/cider/vqa_%s_words%s.p' % (subset, end_suffix)
    cPickle.dump(ngram_words, open(word_file, 'w'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    idx_file = 'data/cider/vqa_%s_idxs%s.p' % (subset, end_suffix)
    cPickle.dump(ngram_idxs, open(idx_file, 'w'),
                 protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train = load_and_process_metadata('train')
    val = load_and_process_metadata('val')
    meta = train + val
    _process_subset(meta, 'kptrain')
    _process_subset(meta, 'kprestval')
