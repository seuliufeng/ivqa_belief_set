import os
from util import get_image_feature_root, load_json, save_json
import numpy as np
from skimage.io import imread
import pylab as plt


def _is_int(x):
    try:
        int(x)
        return True
    except:
        return False


class MCAnnotator(object):
    def __init__(self, subset='val'):
        anno_file = '/import/vision-datasets001/fl302/code/iccv_vaq/data/MultipleChoicesQuestionsKarpathy%sV2.0.json' % subset.title()
        self._subset = subset
        d = load_json(anno_file)
        self._id2type = d['candidate_types']
        self._annotations = d['annotation']
        self.num = len(self._annotations)
        self._im_root = get_image_feature_root()
        self.prog_str = ''

    def set_progress(self, prog_str):
        self.prog_str = prog_str

    def collect_annotation(self, idx):
        info = self._annotations[idx]
        labels = np.array(info['labels'])
        image_id = info['image_id']
        questions = info['questions']
        answer = info['answer']
        answer_idx = info['answer_id']
        # load image
        fname = 'COCO_val2014_%012d.jpg' % image_id
        im_path = os.path.join(self._im_root, 'val2014', fname)
        im = imread(im_path)

        plt.imshow(im)
        plt.draw()

        # print questions

        def print_type(type_id):
            index = np.where(labels == type_id)[0].tolist()
            type_str = self._id2type[str(type_id)]
            for i, idx in enumerate(index):
                print('%s %d: %s' % (type_str, i, questions[idx]))
            return index

        def print_gt():
            print_type(0)
            print(' ')

        def print_head():
            print('=========== %s ===========' % self.prog_str)

        confused = []
        for k in range(5)[1:]:
            os.system('clear')
            print_head()
            print('Answer: %s' % answer)
            print_gt()
            print('Candidates:')
            gindex = print_type(k)
            plt.show(block=False)
            print('\n')
            instruct = '******************************************************************\n' \
                       'Please choose any of the questions holds for this image and answer. \n' \
                       'If any holds, type in the number in front. If not, press enter. If \n' \
                       'multiple questions holders, please seperate with comma, no space.\n' \
                       '******************************************************************\n\n'
            usr_input = raw_input(instruct)
            if usr_input == "":
                continue
            elif ',' in usr_input:
                idx = [int(u) for u in usr_input.split(',')]
                gidx = [gindex[_ix] for _ix in idx]
                confused += gidx
            elif _is_int(usr_input):
                idx = int(usr_input)
                gidx = gindex[idx]
                confused.append(gidx)
            else:
                print(usr_input)

        # verify
        print('\nYou''ve selected:')
        for idx in confused:
            print('%s' % questions[idx])
        raw_input("Press any key to confirm...")
        anno = {'answer_idx': answer_idx, 'confused': confused}
        return anno


class AnnotatorContext(object):
    def __init__(self):
        self.num = 300
        self.annotator = MCAnnotator()
        self.tot_num = self.annotator.num
        self.idx = 0
        self._check_annotation_file()

    def _check_annotation_file(self):
        self.ann_file = 'data/tmp.json'
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
    annotator = AnnotatorContext()
    annotator.run()
    # parse_results()
