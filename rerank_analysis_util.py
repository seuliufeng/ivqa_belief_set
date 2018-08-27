import pdb
from inference_utils.question_generator_util import SentenceGenerator
import numpy as np
import numpy.random as nr


class RerankAnalysiser(object):
    def __init__(self):
        self.labels = []
        self.rerank_preds = []
        self.vqa_top_scores = []
        self.vqa_top_preds = []
        self.vqa_cands = []
        self.to_sentence = SentenceGenerator(trainset='trainval')
        self.file_stream = open('result/rerank_analysis.txt', 'w')

    def update(self, reader_outputs, model_prediction):
        _, _, quest, quest_len, label, _, _, quest_id, image_id = reader_outputs
        score, reranked, vqa_cands, vqa_scores = model_prediction
        # save vqa predictions
        self.vqa_top_preds.append(vqa_cands[:, 0])
        self.vqa_top_scores.append(vqa_scores[:, 0])
        self.vqa_cands.append(vqa_cands)
        # save ivqa predictions
        self.rerank_preds.append(reranked)
        self.labels.append(label)
        self.update_log(quest, quest_len, vqa_cands, vqa_scores, reranked, label,
                        image_id, quest_id)

    def update_log(self, quest, quest_len, vqa_cands, vqa_scores, rerank, label, image_id, quest_id):
        idx = nr.randint(len(quest))
        quest_seq = quest[idx][:quest_len[idx]]
        _log = '-------- image_id: %d, quest_id: %d --------\n' % (image_id[idx], quest_id[idx])
        self.file_stream.write(_log)
        question = self.to_sentence.index_to_question(quest_seq)
        gt_label = label[idx]
        answer = self.to_sentence.index_to_top_answer(label[idx]) if gt_label < 2000 else 'UNK'
        _log = 'Q: %s\n' % question
        self.file_stream.write(_log)
        _log = 'A: %s\n' % answer
        self.file_stream.write(_log)

        r_id = rerank[idx]
        for i, (c_id, c_score) in enumerate(zip(vqa_cands[idx], vqa_scores[idx])):
            cand_answer = self.to_sentence.index_to_top_answer(c_id)
            if c_id == r_id:
                _log = '[%d]: %s (%0.2f)\t<<\n' % (i, cand_answer, c_score)
            else:
                _log = '[%d]: %s (%0.2f)\n' % (i, cand_answer, c_score)
            self.file_stream.write(_log)

    def refine_prediction(self, thresh=0.2):
        rep_tab = self.vqa_top_scores < thresh
        preds = self.vqa_top_preds.copy()
        preds[rep_tab] = self.rerank_preds[rep_tab]
        return preds

    def compute_accuracy(self):
        self.vqa_cands = np.concatenate(self.vqa_cands)
        self.vqa_top_preds = np.concatenate(self.vqa_top_preds)
        self.vqa_top_scores = np.concatenate(self.vqa_top_scores)
        self.labels = np.concatenate(self.labels)
        self.rerank_preds = np.concatenate(self.rerank_preds)

        valid_tab = self.labels < 2000

        def _get_num_col(x):
            if len(x.shape) == 1:
                return 1
            else:
                return x.shape[1]

        def compute_recall(preds, cond_tab=None):
            top_k = _get_num_col(preds)
            preds = preds.reshape([-1, top_k])
            num = preds.shape[0]
            match = np.zeros(num)
            for k in range(top_k):
                pred = preds[:, k]
                match += np.equal(pred, self.labels)
            correct = np.greater(match, 0)
            if cond_tab is None:
                cond_tab = valid_tab
            else:
                cond_tab = np.logical_and(valid_tab, cond_tab)

            valid_correct = correct[cond_tab]
            acc = valid_correct.sum() / float(valid_correct.size)
            prop = cond_tab.sum() / float(valid_tab.sum())
            return acc * 100, prop * 100

        print('\n')
        print('VQA acc@1: %0.2f [%0.1f%%]' % compute_recall(self.vqa_top_preds))
        print('VQA acc@3: %0.2f [%0.1f%%]' % compute_recall(self.vqa_cands))
        print('iVQA acc@1: %0.2f [%0.1f%%]' % compute_recall(self.rerank_preds))
        print('VQA and iVQA acc@1: %0.2f [%0.1f%%]' % compute_recall(self.vqa_top_preds,
                                                           np.equal(self.vqa_top_preds, self.rerank_preds)))
        thresh = np.arange(0.1, 1, 0.1, np.float32)
        for t in thresh:
            acc, p = compute_recall(self.vqa_top_preds, np.greater(self.vqa_top_scores, t))
            print('VQA acc@1 [t=%0.1f]: %0.2f [%0.1f%%]' % (t, acc, p))

        print('\nRefine:')
        thresh = np.arange(0.05, 1, 0.05, np.float32)
        for t in thresh:
            acc, p = compute_recall(self.refine_prediction(t))
            print('Refine VQA acc@1 [t=%0.2f]: %0.2f [%0.1f%%]' % (t, acc, p))

    def close(self):
        self.file_stream.close()
