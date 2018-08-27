with open('data/vqa_trainval_question_word_counts.txt', 'r') as fs:
    lines = fs.readlines()
    words = [line.split()[0].strip() for line in lines]

with open('tmp_dump.txt', 'w') as fs:
    for word in words:
        fs.write('%s\n' % word)

from nltk.corpus import wordnet as wn
import nltk
import numpy as np


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


# generated lemmatized words
lemmatized = []
for i, word in enumerate(words):
    pos_tag = nltk.pos_tag([word])
    tag = pos_tag[0][-1]
    wn_type = penn_to_wn(tag)
    if wn_type is None:
        lem_word = word
    else:
        lem_word = nltk.stem.WordNetLemmatizer().lemmatize(word, wn_type)
    lemmatized.append(lem_word)

# build mapping
vocab = {word: i for i, word in enumerate(words)}

index = []
for lem_word, word in zip(lemmatized, words):
    try:
        id = vocab[lem_word]
    except:
        id = vocab[word]
    index.append(id)
index = np.array(index, dtype=np.int32)

from scipy.io import savemat

savemat('data/quest_token2lemma.mat', {'word2lemma': index})