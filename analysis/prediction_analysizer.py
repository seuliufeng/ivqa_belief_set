import pylab as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import os
from nltk.tokenize import word_tokenize

from inference_utils.question_generator_util import SentenceGenerator
from w2v_answer_encoder import MultiChoiceQuestionManger
from watch_model import mkdir_if_missing


def _tokenize_sentence(sentence):
    return word_tokenize(str(sentence).lower())


class PredictionVisualiser(object):
    def __init__(self, model_name, K=3, do_plot=True):
        self._gt_mgr = MultiChoiceQuestionManger(subset='trainval', load_ans=True)
        self._rev_map = SentenceGenerator(trainset='trainval')
        self._top_k = K
        self._do_plot = do_plot
        self._model_name = model_name
        self._cache_dir = 'att_maps/%s' % self._model_name
        mkdir_if_missing(self._cache_dir)

    def plot(self, quest_id, scores, att_map):
        if type(quest_id) != int:
            quest_id = int(quest_id)
        scores = scores.flatten()
        if scores.size == 2001:
            scores[-1] = 0
        # show question and gt answer
        question = self._gt_mgr.get_question(quest_id)
        gt_ans = self._gt_mgr.get_gt_answer(quest_id)
        print('\n====================================')
        print('Q: %s' % question)
        print('A: %s' % gt_ans)
        # show top k prediction
        index = (-scores).argsort()[:self._top_k]
        for idx in index:
            pred_ans = self._rev_map.index_to_top_answer(idx)
            print('P: %-20s\t(%0.2f)' % (pred_ans, scores[idx]))
        print('\n')
        # show image
        im_file = self._gt_mgr.get_image_file(quest_id)
        im = imread(im_file)
        if np.rank(im) == 2:
            im = np.tile(im[::, np.newaxis], [1, 1, 3])
        if self._do_plot:
            imshow(im)
            plt.show()
        else:
            self.save_cache_file(quest_id, im, att_map, question)
            return
        # show attention map
        tokens = _tokenize_sentence(question)
        self._show_attention_map(im, att_map, tokens)

    def save_cache_file(self, quest_id, im, att_map, question):
        from scipy.io import savemat
        sv_path = os.path.join(self._cache_dir, '%d.mat' % quest_id)
        savemat(sv_path, {'im': im, 'att_map': att_map, 'quest': question})

    def _show_attention_map(self, im, att_map, tokens):
        att_map = att_map.reshape([-1, 14, 14])
        num = att_map.shape[0]
        if num == 1:
            tokens = [' '.join(tokens)]  # merge to a sentence
        else:
            tokens = [' '.join(tokens)]  # merge to a sentence
            # mean_map = att_map.mean(axis=0)[np.newaxis, ::]
            # att_map = np.concatenate([att_map, mean_map], axis=0)
            # tokens.append('average')
        # render and plot
        for i, am in enumerate(att_map):
            am = resize(am, im.shape[:2], preserve_range=True)
            am = am / am.max()
            v = im * am[:, :, np.newaxis]
            v = np.minimum(np.round(v).astype(np.uint8), 255)
            if self._do_plot:
                imshow(v)
                plt.title('%s <%d>' % (tokens[0], i))
                plt.show()


if __name__ == '__main__':
    import numpy.random as nr
    from scipy.io import loadmat

    # d = loadmat('data/vis_debug_data.mat')
    # qid = int(d['qid'])
    # scores = d['scores']
    # att_map = d['att_map']
    # pm = PredictionVisualiser(3)
    # pm.plot(qid, scores, att_map)
