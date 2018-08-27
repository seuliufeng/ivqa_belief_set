import os
from util import get_image_feature_root, load_json, save_json
import numpy as np
from skimage.io import imread
import pylab as plt
from w2v_answer_encoder import MultiChoiceQuestionManger


def _is_int(x):
    try:
        int(x)
        return True
    except:
        return False


class MCAnnotator(object):
    def __init__(self, result_file, subset='val'):
        self._subset = subset
        self.results = load_json(result_file)
        self.num = len(self.results)
        self._im_root = get_image_feature_root()
        self.prog_str = ''
        self.mc_ctx = MultiChoiceQuestionManger(subset='val')

    def set_progress(self, prog_str):
        self.prog_str = prog_str

    def collect_annotation(self, idx):
        info = self.results[idx]
        ratings = ['Perfect', 'Correct', 'Wrong']

        # get info
        image_id = info['image_id']
        question_id = info['question_id']
        question = info['question']
        answer = self.mc_ctx.get_gt_answer(question_id)

        # load image
        fname = 'COCO_val2014_%012d.jpg' % image_id
        im_path = os.path.join(self._im_root, 'val2014', fname)
        im = imread(im_path)

        plt.imshow(im)
        plt.draw()

        # print questions

        def print_head():
            print('=========== %s ===========' % self.prog_str)

        os.system('clear')
        print_head()

        while True:
            print('Question: %s' % question)
            print('Answer: %s' % answer)
            plt.show(block=False)

            instruct = '******************************************************************\n' \
                       'Please choose any of the questions holds for this image and answer. \n' \
                       'If any holds, type in the number in front. If not, press enter. If \n' \
                       'multiple questions holders, please seperate with comma, no space.\n' \
                       '******************************************************************\n\n'
            usr_input = raw_input(instruct)

            if _is_int(usr_input):
                r_idx = int(usr_input)
                if r_idx >= 3:
                    print('Should be in [0, 1, 2]')
                    continue
            else:
                print('illegal input, choose again')
                continue

            # verify
            r_str = ratings[r_idx]
            print('\nYour rating is %d [%s]:' % (r_idx, r_str))
            usr_input = raw_input("Press c to confirm, r to undo...")
            if usr_input == 'c':
                break
            else:
                continue

        anno = {'question_id': question_id, 'rating': r_idx}
        return anno


class AnnotatorContext(object):
    def __init__(self, result_file):
        self.num = 300
        self.annotator = MCAnnotator(result_file)
        self.tot_num = self.annotator.num
        self.idx = 0
        self._check_annotation_file()

    def _check_annotation_file(self):
        self.ann_file = 'data/human_study.json'
        if os.path.exists(self.ann_file):  # restore
            print('Loading annotation file')
            d = load_json(self.ann_file)
            self.index_processed = d['index_processed']
            self.index_to_proc = d['index_to_proc']
            self.annotation = d['annotation']
        else:
            print('Creating new annotation file')
            self.index_processed = []
            index = np.arange(self.tot_num)
            np.random.shuffle(index)
            self.index_to_proc = index[:self.num].tolist()
            self.annotation = []
            self.index_processed = []
            self._update_file()

    def _update_file(self):
        d = {}
        d['index_to_proc'] = self.index_to_proc[self.idx:]
        d['index_processed'] = self.index_processed
        d['annotation'] = self.annotation
        save_json(self.ann_file, d)

    def get_progress(self):
        num_proc = len(self.index_processed)
        return '%d/%d' % (num_proc + 1, self.num)

    def run(self):
        for idx in self.index_to_proc:
            # launch annotation
            self.annotator.set_progress(self.get_progress())
            anno = self.annotator.collect_annotation(idx)
            self.annotation.append(anno)
            # move pointer
            self.index_processed.append(idx)
            self.idx += 1
            # update file
            self._update_file()


def parse_results():
    ann_file = 'data/tmp.json'
    anno = load_json(ann_file)['annotation']
    num_confused = np.array([len(info['confused']) for info in anno], dtype=np.float32)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    res_file = '/import/vision-datasets001/fl302/code/iccv_vaq/result_hist/beamsearch_vaq_VAQ-LSTM-DEC-SUP_kpval.json'
    annotator = AnnotatorContext(result_file=res_file)
    annotator.run()
    # parse_results()
