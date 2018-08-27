from util import *
from generate_new_metric_candidates import QuestionVocab
import pylab as plt
import os


class MultiChoiceConfig(object):
    def __init__(self):
        self.num_corr_or_contrast = 6
        self.num_plausible = 6
        self.num_popular = 6
        self.num_total = 24
        self.type = {'correct': 0,
                     'contrast': 1,
                     'plausible': 2,
                     'popular': 3,
                     'random': 4}
        self.id2type = {0: 'GT',
                        1: 'CT',
                        2: 'PS',
                        3: 'PP',
                        4: 'RN'}

    def get_type_id(self, type):
        return self.type[type]

    def id_to_type(self, id):
        return self.id2type[id]


def _add_unique_questions(cands, selected, labels, type_id, num=None):
    if num is None:
        num = len(cands)

    idx = 0
    for c in cands:
        if c in selected:
            continue
        else:
            selected.append(c)
            labels.append(type_id)
            idx += 1
        if idx >= num:
            break
    return selected, labels


def _process_sample(info, config, quest_vocab):
    quest_index, labels = [], []

    # add gt questions
    correct_index = info['correct_question_index']
    num_correct = correct_index.size
    quest_index, labels = _add_unique_questions(info['correct_question_index'],
                                                quest_index,
                                                labels,
                                                config.get_type_id('correct'))

    # add contrastive questions
    num_contrast = config.num_corr_or_contrast - num_correct
    quest_index, labels = _add_unique_questions(info['contrastive_question_index'],
                                                quest_index,
                                                labels,
                                                config.get_type_id('contrast'),
                                                num_contrast)

    # add plausible questions
    quest_index, labels = _add_unique_questions(info['plausible_question_index'][10:],
                                                quest_index,
                                                labels,
                                                config.get_type_id('plausible'),
                                                config.num_plausible)

    # add popular questions
    quest_index, labels = _add_unique_questions(info['popular_question_index'],
                                                quest_index,
                                                labels,
                                                config.get_type_id('popular'),
                                                config.num_popular)

    # add random questions
    num_rand = config.num_total - len(quest_index)
    quest_index, labels = _add_unique_questions(info['random_question_index'],
                                                quest_index,
                                                labels,
                                                config.get_type_id('random'),
                                                num_rand)

    # shuffle
    num = len(quest_index)
    order = np.random.permutation(num)
    labels = [int(labels[idx]) for idx in order]
    quest_index = [int(quest_index[idx]) for idx in order]
    questions = [quest_vocab.index_to_question_key(idx) for idx in quest_index]

    # create data structure for output
    datum = {'image_id': int(info['image_id']),
             'answer_id': int(info['answer_id']),
             'answer': info['answer'],
             'labels': labels,
             'questions': questions,
             'question_ids': quest_index,
             'coco_question_ids': info['correct_question_ids']}
    return datum


def print_annotation(info, config, show_image=False):
    answer_id = info['answer_id']
    print('\n==================== %d ===================' % answer_id)
    print('A: %s' % info['answer'])
    for i, (quest, quest_type_id) in enumerate(zip(info['questions'], info['labels'])):
        print('%02d: [%s] %s' % (i, config.id_to_type(quest_type_id).upper(),
                                 quest.capitalize()))
    print('\n')
    if show_image:
        im_root = get_image_feature_root()
        im_name = 'val2014/COCO_val2014_%012d.jpg' % info['image_id']
        im_path = os.path.join(im_root, im_name)
        im = plt.imread(im_path)
        plt.imshow(im)
        plt.show()


def generate_candidates():
    subset = 'test'
    config = MultiChoiceConfig()
    info_file = 'data/ivqa_multiple_choices_test_questions.pkl'
    d = unpickle(info_file)
    dataset = d['dataset']
    quest_vocab = d['quest_vocab']

    annotations = []
    for i, datum in enumerate(dataset):
        if i % 100 == 0:
            print('Metric Maker: generate %d/%d samples' % (i, len(dataset)))
        ann = _process_sample(datum, config, quest_vocab)
        annotations.append(ann)
        if i % 1000 == 0:
            print_annotation(ann, config)
    print('Metric Maker: saving results, %d samples generated' % len(annotations))
    res_file = 'data/MultipleChoicesQuestionsKarpathy%s.json' % subset.title()
    save_json(res_file, {'annotation': annotations,
                         'candidate_types': config.id2type})


if __name__ == '__main__':
    generate_candidates()
