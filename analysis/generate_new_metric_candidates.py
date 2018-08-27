import numpy as np
from util import *
from w2v_answer_encoder import MultiChoiceQuestionManger
from w2v_answer_encoder import _tokenize_sentence
from build_karpathy_split_data import get_image_id
from scipy.spatial.distance import cdist


def load_dataset():
    def _read_json(data_root, filename, key=None):
        d = json.load(open(os.path.join(data_root,
                                        filename), 'r'))
        return d if key is None else d[key]

    ann_root = 'data/annotations'
    questions = _read_json(ann_root, 'MultipleChoice_mscoco_val2014_questions.json', 'questions')
    annotations = _read_json(ann_root, 'mscoco_val2014_annotations.json', 'annotations')
    return questions, annotations


def _generate_key(sentence):
    tokenized = _tokenize_sentence(sentence)
    key = ' '.join(tokenized)
    return key


def build_candidate_answers(answer_dict, split='val'):
    print('Metric Maker: preparing data split %s ...' % split)
    coco_image_id = get_image_id(split)
    subset_dict = dict([(idx, 0) for idx in coco_image_id])
    _, annotations = load_dataset()
    inv_answer_dict = {v['answer_idx']: k for k, v in answer_dict.iteritems()}

    # find corresponding question and answers
    buf = {}
    for info in annotations:
        image_id = info['image_id']
        if image_id not in subset_dict:  # skip
            continue
        # append this sample
        quest_id = info['question_id']
        mc_answer = info['multiple_choice_answer']
        mc_answer_key = _generate_key(mc_answer)
        answer_idx = answer_dict[mc_answer_key]['answer_idx']
        if image_id not in buf:
            buf[image_id] = {answer_idx: [quest_id]}
        else:
            if answer_idx in buf[image_id]:
                buf[image_id][answer_idx].append(quest_id)
            else:
                buf[image_id][answer_idx] = [quest_id]

    # reorganise the data
    dataset = []
    for image_id in coco_image_id:
        for i, idx in enumerate(buf[image_id]):
            answer_id_str = '%d%01d' % (image_id, i)
            answer_id = int(answer_id_str)
            quest_id = buf[image_id][idx]
            info = {'answer_id': answer_id,
                    'answer': inv_answer_dict[idx],
                    'correct_question_ids': quest_id,
                    'image_id': image_id}
            dataset.append(info)
    print('Metric Maker: loaded %d unique answer-image pairs\n' % len(dataset))
    return dataset


def add_popular_questions(dataset, mc_ctx, quest_vocab,
                          quest_index_by_answer_type, num=6):
    for i, info in enumerate(dataset):
        if i % 1000 == 0:
            print('Metric Maker: adding popular questions %d/%d' % (i, len(dataset)))
        cands = {}
        correct_index = []
        for quest_id in info['correct_question_ids']:
            answer_type = mc_ctx.get_answer_type(quest_id)
            tmp_index = quest_index_by_answer_type[answer_type][:num + 1]
            cands[answer_type] = np.setdiff1d(tmp_index, correct_index)
            correct_index.append(quest_vocab.question_id2index(quest_id))
        correct_index = np.array(correct_index)
        if len(cands) > 1:  # if one answer corresponds to multiple answer types
            answer_id = info['answer_id']
            print('Waring: answer %d has multiple types' % answer_id)
            cands = np.concatenate(cands.values(), axis=0).flatten(order='F')[:num]
        else:  # if it only corresponds to one single answer type
            cands = cands.values()[0][:num]
        info['correct_question_index'] = correct_index
        info['popular_question_index'] = cands
    return dataset


class VisualFeatureVocab(object):
    def __init__(self, exclude_self=False, l2_normalise=True, EPS=1e-12):
        print('VisualFeatureVocab: loading and processing features...')
        d = load_hdf5('data/vgg19_mscoco_val2014.h5')
        vis_feats = d['feats']
        image_ids = d['image_ids']
        self._image_id2index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self._index2image_id = {idx: image_id for idx, image_id in enumerate(image_ids)}
        self._vis_feature = vis_feats
        if l2_normalise:
            l2_norm = np.sqrt(np.square(self._vis_feature).sum(axis=1)[:, np.newaxis])
            self._vis_feature = np.divide(self._vis_feature, l2_norm + EPS)
            print('VisualFeatureVocab: feature l2 normalised\n')
        self._exclude_self = exclude_self

    def get_nearest_image_ids(self, image_id, K):
        query_idx = self._image_id2index[image_id]
        query_feat = self._vis_feature[query_idx][np.newaxis, :]
        d2 = cdist(query_feat, self._vis_feature).flatten()
        assert (d2[query_idx] == 0.0)
        if self._exclude_self:
            nn_index = d2.argsort()[1:K + 1]
        else:
            nn_index = d2.argsort()[:K]
        return [self._index2image_id[idx] for idx in nn_index]


def add_contrastive_questions(dataset, mc_ctx, quest_vocab, num=100):
    # create visual encoder
    vis_vocab = VisualFeatureVocab()
    # organise data by images
    image2info = {}
    for info in dataset:
        image_id = info['image_id']
        image2info.setdefault(image_id, []).append(info)
    # process dataset
    for i, (image_id, answer_list) in enumerate(image2info.iteritems()):
        if i % 100 == 0:
            print('Metric Maker: adding contrastive questions %d/%d' % (i, len(image2info)))
        nn_image_ids = vis_vocab.get_nearest_image_ids(image_id, num * 4)
        for ans_tuple in answer_list:
            # get answer types for this answer
            answer_type = [mc_ctx.get_answer_type(quest_id) for quest_id in ans_tuple['correct_question_ids']]
            # get all candidate questions
            cand_quest_ids = [quest_id for nn_im_id in nn_image_ids for quest_id in mc_ctx.image_id2quest_ids(nn_im_id)]
            contrast_quest_ids = []
            # filter by answer type (choose from different type)
            for quest_id in cand_quest_ids:
                if mc_ctx.get_answer_type(quest_id) in answer_type:
                    continue
                else:
                    contrast_quest_ids.append(quest_id)
                if len(contrast_quest_ids) == num:
                    break
            ans_tuple['contrastive_question_id'] = contrast_quest_ids
            ans_tuple['contrastive_question_index'] = [quest_vocab.question_id2index(quest_idx) for quest_idx in
                                                       contrast_quest_ids]
        if i % 100 == 0:  # print every 1000 iterations
            print('\n============ contrastive questions ===========')
            sampled_quest_id = ans_tuple['correct_question_ids'][0]
            print('GT: %s' % (quest_vocab.question_id_to_question_key(sampled_quest_id).capitalize()))
            for k, ident in enumerate(ans_tuple['contrastive_question_index'][:20]):
                print('%02d: %s' % (k + 1, quest_vocab.index_to_question_key(ident).capitalize()))
            print('\n')
    return dataset


class QuestionFeatureVocab(object):
    def __init__(self, quest_vocab, exclude_self=True, split_feature=True):
        # loading data
        d = load_hdf5('data/vqa_val_skipthought.h5')
        quest_ids = d['quest_id']
        quest_features = d['quest_coding']
        # change the key to quest indent and remove duplicated
        tmp_quest_ident2mat_index = {}
        for i, quest_id in enumerate(quest_ids):
            quest_ident = quest_vocab.question_id2index(quest_id)
            if quest_ident not in tmp_quest_ident2mat_index:
                tmp_quest_ident2mat_index[quest_ident] = i
        # find unique question coding and slice
        indents = quest_vocab.get_index()
        valid_row = np.array([tmp_quest_ident2mat_index[idx] for idx in indents], dtype=np.int32)
        # set variables
        self._split_feature = split_feature
        self._quest_vocab = quest_vocab
        self._quest_feature = quest_features[valid_row, :]  # stored according to index of quest vocab
        if self._split_feature:
            self._quest_feature = [self._quest_feature[:, :2400], self._quest_feature[:, 2400:]]
        else:
            self._quest_feature = [self._quest_feature]
        self._exclude_self = exclude_self

    def _compute_distance(self, query_idx):
        d = 0.0
        for quest_feature in self._quest_feature:
            d += cdist(quest_feature[query_idx][np.newaxis, :], quest_feature).flatten()
        return d

    def get_nearest_question_index_from_index(self, query_idx, K):
        d2 = self._compute_distance(query_idx)
        assert (d2[query_idx] == 0.0)
        if self._exclude_self:
            nn_index = d2.argsort()[1:K + 1]
        else:
            nn_index = d2.argsort()[:K]
        return nn_index

    def get_nearest_question_index_from_quest_id(self, quest_id, K):
        query_ident = self._quest_vocab.question_id2index(quest_id)
        return self.get_nearest_question_index_from_index(query_ident, K)


def add_plausible_questions(dataset, quest_vocab, num=100):
    quest_ctx = QuestionFeatureVocab(quest_vocab, exclude_self=True)
    for i, datum in enumerate(dataset):
        if i % 100 == 0:
            print('Metric Maker: adding plausible questions %d/%d' % (i, len(dataset)))

        quest_ids = []
        for quest_id in datum['correct_question_ids']:
            this_num = int(num / len(datum['correct_question_ids'])) + 1
            quest_idx = quest_ctx.get_nearest_question_index_from_quest_id(quest_id,
                                                                           this_num)
            quest_ids.append(quest_idx[np.newaxis, :])
        datum['plausible_question_index'] = np.concatenate(quest_ids).flatten(order='F')[:num]
        # print plausible questions, print every 1000 iterations
        if i % 1000 == 0:
            print('\n============ plausible questions ===========')
            print('GT: %s' % (quest_vocab.question_id_to_question_key(quest_id).capitalize()))
            for k, ident in enumerate(datum['plausible_question_index'][:20]):
                print('%02d: %s' % (k + 1, quest_vocab.index_to_question_key(ident).capitalize()))
            print('\n')
    return dataset


def add_random_questions(dataset, mc_ctx, quest_vocab, answer_dict, quest_index_by_answer_type, num=100):
    for i, datum in enumerate(dataset):
        answer = datum['answer']
        answer_key = _generate_key(answer)
        quest_ids = answer_dict[answer_key]['quest_id']  # this is question id
        # This is 5:51 a.m. in the morning, just to work this ugly code out.
        # I'm fucking high, Guinness is really good. Back to wor ... k, where have I got
        correct_ids = np.array(datum['correct_question_ids'])
        valid_quest_index = np.array([quest_vocab.question_id2index(quest_id) for quest_id
                                      in np.setdiff1d(quest_ids, correct_ids)], dtype=np.int32)
        if valid_quest_index.size > num:
            rand_quest_ids = np.random.choice(valid_quest_index, size=(num,), replace=False)
        else:  # not having enough questions for this particular answer, just sample from the same answer type
            quest_index_step0 = valid_quest_index
            num_to_sample = num - valid_quest_index.size
            correct_quest_id = quest_ids[0]
            answer_type = mc_ctx.get_answer_type(correct_quest_id)
            # sample slightly more, counting for intersection
            quest_index_step1 = np.random.choice(quest_index_by_answer_type[answer_type], num_to_sample * 2)
            quest_index_step1 = np.setdiff1d(quest_index_step1, quest_index_step0)
            rand_quest_ids = np.concatenate([quest_index_step0, quest_index_step1])
            rand_quest_ids = rand_quest_ids[:min(num, rand_quest_ids.size)]
        datum['random_question_index'] = rand_quest_ids
        # print plausible questions, print every 1000 iterations
        if i % 1000 == 0:
            print('\n============ Random questions ===========')
            quest_id = correct_ids[0]
            print('GT: %s' % (quest_vocab.question_id_to_question_key(quest_id).capitalize()))
            for k, ident in enumerate(datum['random_question_index'][:20]):
                print('%02d: %s' % (k + 1, quest_vocab.index_to_question_key(ident).capitalize()))
            print('\n')
    return dataset


class QuestionVocab(object):
    def __init__(self, question_dict, question_id2question_key):
        self._quest_id2quest_key = question_id2question_key
        self._quest_key2index = {qk: i for i, qk in enumerate(question_dict.keys())}
        self._index2quest_key = {i: qk for i, qk in enumerate(question_dict.keys())}
        self._index = np.arange(len(question_dict), dtype=np.int32)

    def question_id2index(self, quest_id):
        key = self._quest_id2quest_key[quest_id]
        return self._quest_key2index[key]

    def index_to_question_key(self, index):
        return self._index2quest_key[index]

    def question_id_to_question_key(self, quest_id):
        return self.index_to_question_key(self.question_id2index(quest_id))

    def get_question_ids(self):
        return self._quest_id2quest_key.keys()

    def get_index(self):
        return self._index


def sort_questions_by_answer_type(mc_ctx, quest_vocab,
                                  max_keep=1000):
    ans_type_dict = {}
    quest_ids = quest_vocab.get_question_ids()
    for i, quest_id in enumerate(quest_ids):
        if i % 1000 == 0:
            print('Metric Maker: parsing answer types %d/%d' % (i, len(quest_ids)))
        ans_type = mc_ctx.get_answer_type(quest_id)
        quest_key_ident = quest_vocab.question_id2index(quest_id)
        if ans_type in ans_type_dict:
            ans_type_dict[ans_type].append(quest_key_ident)
        else:
            ans_type_dict[ans_type] = [quest_key_ident]
    # find unique keys and sort
    quest_key_idents_per_answer_type = {}
    for ans_type in ans_type_dict:
        quest_idents = np.array(ans_type_dict[ans_type])
        counts = np.bincount(quest_idents)
        ordered_idents = (-counts).argsort()[:max_keep]
        quest_key_idents_per_answer_type[ans_type] = ordered_idents
        print('\n=========== %s popular questions ===========' % ans_type.upper())
        for k, ident in enumerate(ordered_idents[:20]):
            print('%02d: %s (%d)' % (k + 1, quest_vocab.index_to_question_key(ident).capitalize(), counts[ident]))
    return quest_key_idents_per_answer_type


def main():
    split = 'test'
    data_file = 'data/ivqa_multiple_choices_%s_questions.pkl' % split
    print(data_file)
    mc_ctx = MultiChoiceQuestionManger(subset='val', load_ans=True)
    # find unique questions
    # questions = load_questions()
    question_ids = mc_ctx.get_question_ids()

    question_dict = {}
    answer_dict = {}
    question_id2question_key = {}

    # set question and answer keys
    unique_question_idx = 0
    answer_idx = 0
    for i, quest_id in enumerate(question_ids):
        if i % 1000 == 0:
            print('Metric Maker: parsed %d/%d questions' % (i, len(question_ids)))
        question = mc_ctx.get_question(quest_id)
        quest_key = _generate_key(question)
        question_id2question_key[quest_id] = quest_key
        if quest_key in question_dict:
            question_dict[quest_key]['counts'] += 1
        else:
            question_dict[quest_key] = {'counts': 1, 'key_idx': unique_question_idx}
            unique_question_idx += 1
        # parse answers
        mc_answer = mc_ctx.get_gt_answer(quest_id)
        answer_key = _generate_key(mc_answer)

        if answer_key in answer_dict:
            answer_dict[answer_key]['quest_id'].append(quest_id)
        else:
            answer_dict[answer_key] = {'quest_id': [quest_id],
                                       'answer_idx': answer_idx}
            answer_idx += 1
    # sort questions by answer type
    quest_vocab = QuestionVocab(question_dict, question_id2question_key)
    quest_index_by_answer_type = sort_questions_by_answer_type(mc_ctx, quest_vocab)

    # build basic data structure for iVQA
    dataset = build_candidate_answers(answer_dict, split=split)

    # add popular questions
    dataset = add_popular_questions(dataset, mc_ctx, quest_vocab,
                                    quest_index_by_answer_type)

    # add contrastive questions
    dataset = add_contrastive_questions(dataset, mc_ctx, quest_vocab, num=100)

    # add plausible questions
    dataset = add_plausible_questions(dataset, quest_vocab, num=100)

    # add random questions
    dataset = add_random_questions(dataset, mc_ctx, quest_vocab,
                                   answer_dict, quest_index_by_answer_type, num=200)

    # save data
    data_file = 'data/ivqa_multiple_choices_%s_questions.pkl' % split
    pickle(data_file, {'dataset': dataset, 'quest_vocab': quest_vocab})


if __name__ == '__main__':
    main()
