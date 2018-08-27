# import tensorflow as tf
from nltk.stem.wordnet import WordNetLemmatizer
from extract_vqa_word2vec_coding import load_and_process_metadata, split_data_by_seed
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from util import save_json
import pdb


def build_question_set():
    sv_file = 'data/kprestval_pos_tags.json'
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    meta = load_and_process_metadata('val')
    images = split_data_by_seed(meta, 'kprestval')
    num = len(images)
    pos_tags_dict = {}
    for i, info in enumerate(images):
        question_id = info.question_id
        question = info.question.lower()
        _pos_tags = st.tag(word_tokenize(question))
        pos_tags_dict[question_id] = _pos_tags
        print('\nPOS TAGGER: %d/%d' % (i, num))
        print(_pos_tags)
    save_json(sv_file, {'pos_tags': pos_tags_dict})
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    # WordNetLemmatizer().lemmatize(word, 'v')
    # pass


if __name__ == '__main__':
    build_question_set()
