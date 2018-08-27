from util import load_json, save_json
import re
import numpy as np

_ANNO_FILE = 'data/my_val2014_vqa_anno.json'


def processPunctuation(self, inText):
    outText = inText
    for p in self.punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = self.periodStrip.sub("",
                                   outText,
                                   re.UNICODE)
    return outText


def processDigitArticle(self, inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = self.manualMap.setdefault(word, word)
        if word not in self.articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in self.contractions:
            outText[wordId] = self.contractions[word]
    outText = ' '.join(outText)
    return outText


def convert_vqa_annotations():
    path_src = '../../data/VQA/Annotations/mscoco_val2014_annotations.json'
    d = load_json(path_src)
    anno = {}
    for info in d['annotations']:
        quest_id = int(info['question_id'])
        tmp = {}
        for _a in info['answers']:
            a = _a['answer']
            if a in tmp:
                tmp[a] += 1.0
            else:
                tmp[a] = 1.0
        anno[quest_id] = tmp
    save_json(_ANNO_FILE, anno)


def eval_accuracy(results):
    over_complete = load_json(_ANNO_FILE)
    num_correct = 0
    accs = []
    for res in results:
        quest_id = str(res['question_id'])
        cand = str(res['answer'])
        gts = over_complete[quest_id]
        if cand in gts:
            _n = gts[cand]
        else:
            _n = 0
        accs.append(min(1., float(_n) / 3))
        # num_correct += (_n >= 3)
    num_tot = len(results)
    # mean_acc = num_correct / float(num_tot)
    mean_acc = np.array(accs).mean()
    print('Evaluated %d questions' % num_tot)
    print('Accuracy: %0.2f' % (100. * mean_acc))


def eval_recall(results):
    over_complete = load_json(_ANNO_FILE)
    accs = []
    num_cands = 0
    for res in results:
        quest_id = str(res['question_id'])
        cands = res['answers']
        gts = over_complete[quest_id]
        cand_acc = []
        for cand in cands:
            cand = str(cand)
            cand = cand.strip()
            if cand in gts:
                _n = gts[cand]
            else:
                _n = 0
            cand_acc.append(min(1., float(_n) / 3))
        v = max(cand_acc)
        accs.append(v)
        num_cands += len(cands)
        # num_correct += (_n >= 3)
    num_tot = len(results)
    # mean_acc = num_correct / float(num_tot)
    mean_acc = np.array(accs).mean()
    print('Evaluated %d questions' % num_tot)
    print('Total number of candidates: %d (%0.2f/image)' % (num_cands, float(num_cands)/num_tot))
    print('Recall: %0.2f' % (100. * mean_acc))


def test_my_eval():
    res_file = 'result/v1_vqa_OpenEnded_mscoco_dev2015_baseline_results.json'
    results = load_json(res_file)
    from vqa_eval import evaluate_model
    # quest_ids = [res['question_id'] for res in results]
    # evaluate_model(res_file, quest_ids,
    #                version='v1')
    eval_accuracy(results)


if __name__ == '__main__':
    convert_vqa_annotations()
    test_my_eval()
