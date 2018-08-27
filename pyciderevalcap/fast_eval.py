__author__ = 'rama'
from tokenizer.ptbtokenizer import PTBTokenizer
# from cider.cider import Cider
from ciderD.ciderD import CiderD


class CIDErEvalCap:
    def __init__(self, df):
        # if 'idxs' in df:
        #     _gts = gts
        #     _res = res
        # else:
        #     print 'tokenization...'
        #     tokenizer = PTBTokenizer('gts')
        #     _gts = tokenizer.tokenize(gts)
        #     print 'tokenized refs'
        #     tokenizer = PTBTokenizer('res')
        #     _res = tokenizer.tokenize(res)
        #     print 'tokenized cands'
        #
        # self.gts = _gts
        # self.res = _res
        self.df = df
        self.scorer = CiderD(df=self.df)

    def evaluate(self, gts, res):
        score, scores = self.scorer.compute_score(gts, res)
        return score, scores

    def method(self):
        return "Cider"
