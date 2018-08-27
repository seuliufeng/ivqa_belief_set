import numpy as np
from util import load_json


def process(method):
    res_file = 'result/bs_vis_scores_%s.json' % method
    results = load_json(res_file)
    d = {}
    for item in results:
        quest_id = item['question_id']
        question = item['question']
        score = item['score']
        datum = (question, score)
        if quest_id in d:
            d[quest_id].append(datum)
        else:
            d[quest_id] = [datum]

    log_file = 'result/vis_%s.txt' % method
    with open(log_file, 'w') as fs:
        for quest_id, cands in d.items():
            scores = np.array([c[1] for c in cands])
            questions = np.array([c[0] for c in cands])
            order = (-scores).argsort()
            msg = '\nquestion_id: %d' % quest_id
            fs.write('%s\n' % msg)
            print(msg)
            for idx in order:
                msg = '%s (%0.2f)' % (questions[idx], scores[idx])
                print('%s' % msg)
                fs.write('%s\n' % msg)


def main():
    methods = ['deeperlstm', 'mlb2-att', 'n2mn', 'mlb-att']
    for method in methods:
        process(method)


if __name__ == '__main__':
    main()


