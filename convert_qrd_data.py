import os
import numpy as np
from nltk.tokenize import word_tokenize
from inference_utils import vocabulary
from util import load_json, find_image_id_from_fname
from util import save_hdf5, save_json

def _tokenize_sentence(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    return word_tokenize(str(sentence).lower())


class SentenceEncoder(object):
    def __init__(self, type='question'):
        assert (type in ['answer', 'question'])
        self._vocab = None
        vocab_file = 'data/vqa_%s_%s_word_counts.txt' % ('trainval', type)
        self._load_vocabulary(vocab_file)

    def _load_vocabulary(self, vocab_file=None):
        print('Loading answer words vocabulary...')
        print(vocab_file)
        self._vocab = vocabulary.Vocabulary(vocab_file)

    def encode_sentence(self, sentence):
        return self._encode_sentence(sentence)

    def _encode_sentence(self, sentence):
        tokens = _tokenize_sentence(sentence)
        return [self._vocab.word_to_id(word) for word in tokens]


def make_blacklist():
    d = load_json('data/vqa_std_mscoco_kptest.meta')
    # question_ids = d['quest_id']
    images = d['images']
    image_ids = [find_image_id_from_fname(im_name) for im_name in images]
    return {image_id: None for image_id in image_ids}


def load_qrpe_data(blacklist):
    fdir = '/usr/data/fl302/code/premise17/question_relevance/models/relevance_prediction/qc_sim_models'

    def _load_subset(subset):
        fpath = os.path.join(fdir, 'input/qrpe_%s.json' % subset)
        d = load_json(fpath)
        dataset = []
        db_type = 'train' if subset == 'train' else 'val'
        for item in d.values():
            i_from = item['imtype']
            i_label = int(item['label'])
            if i_label == 2 and 'coco' in i_from:
                image_id = find_image_id_from_fname(item['image'])
                dataset.append({'image': '%s2014/%s' % (db_type, item['image']),
                                'image_id': image_id,
                                'question': item['question']})
        return dataset

    train = _load_subset('train')
    val = _load_subset('test')
    meta = train + val

    # filter out blacklist
    meta_new = [item for item in meta if item['image_id'] not in blacklist]
    return meta_new


def load_vtfp_data(blacklist):
    fdir = '/usr/data/fl302/code/VQARelevance/questionCaptionMatchModels'

    def _load_subset():
        fpath = os.path.join(fdir, 'unique_vtfq.json')
        d = load_json(fpath)
        dataset = []
        for item in d.values():
            i_label = int(item['label'])
            if i_label == 2:  # irrelevant
                image = os.path.basename(item['image'])
                image_id = find_image_id_from_fname(image)
                dataset.append({'image': 'val2014/' + image,
                                'image_id': image_id,
                                'question': item['question'] + '?'})
        return dataset

    meta = _load_subset()
    # filter out blacklist
    meta_new = [item for item in meta if item['image_id'] not in blacklist]
    return meta_new


def load_bsir_dataset():
    fdir = '/usr/data/fl302/code/VQARelevance/questionCaptionMatchModels/'

    def _load_subset():
        fpath = os.path.join(fdir, 'bsir.json')
        d = load_json(fpath)
        dataset = []
        for item in d.values():
            label = item['type'] != 'irrelevant'
            image = item['image']
            image_id = item['image_id']
            dataset.append({'image': 'val2014/' + image,
                            'image_id': image_id,
                            'question': item['target'],
                            'label': label})
        return dataset

    meta = _load_subset()
    return meta


def process():
    # load data
    blacklist = make_blacklist()
    save_json('data/kptest_blacklist.json', blacklist)
    qrpe = load_qrpe_data(blacklist)
    vtfp = load_vtfp_data(blacklist)
    import pdb
    pdb.set_trace()
    meta = qrpe + vtfp
    # process data
    images, image_ids, questions = [], [], []
    encoder = SentenceEncoder()
    for item in meta:
        image_id = item['image_id']
        image = item['image']
        tokens = encoder.encode_sentence(item['question'])
        images.append(image)
        image_ids.append(image_id)
        questions.append(tokens)
    # put to array
    from post_process_variation_questions import put_to_array
    arr, arr_len = put_to_array(questions)

    save_json('data/QRD_irrelevant_meta.json', {'images': images,
                                                'image_ids': image_ids})
    image_ids = np.array(image_ids, dtype=np.int32)
    save_hdf5('data/QRD_irrelevant_data.data', {'image_ids': image_ids,
                                                'quest': arr,
                                                'quest_len': arr_len})


def process_test():
    from util import save_hdf5, save_json
    # load data
    meta = load_bsir_dataset()
    # process data
    labels, images, image_ids, questions = [], [], [], []
    encoder = SentenceEncoder()
    for item in meta:
        image_id = item['image_id']
        image = item['image']
        tokens = encoder.encode_sentence(item['question'])
        images.append(image)
        image_ids.append(image_id)
        questions.append(tokens)
        labels.append(item['label'])
    # put to array
    from post_process_variation_questions import put_to_array
    arr, arr_len = put_to_array(questions)

    save_json('data/QRD_irrelevant_meta_test.json', {'images': images,
                                                     'image_ids': image_ids})
    image_ids = np.array(image_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    import pdb
    pdb.set_trace()
    save_hdf5('data/QRD_irrelevant_data_test.data', {'image_ids': image_ids,
                                                     'quest': arr,
                                                     'quest_len': arr_len,
                                                     'labels': labels})


if __name__ == '__main__':
    # process()
    # process_test()
    blacklist = make_blacklist()
    save_json('data/kptest_blacklist.json', blacklist)
