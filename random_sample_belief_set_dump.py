import numpy as np


def _parse_line(line):
    quest_id = None
    patt = 'question_id: '
    if patt in line:
        quest_id = int(line.replace(patt, '').strip())
        line = None
    return quest_id, line


def random_pick(sampels, num_keep):
    import numpy as np
    perm = np.arange(len(sampels))
    np.random.shuffle(perm)
    return [sampels[i] for i in perm[:num_keep]]


def sort_samples(samples):
    scores = []
    for line in samples:
        score = line.split('?')[-1].strip()
        # remove blankets
        score = score.replace('(', '').replace(')', '')
        score = float(score)
        scores.append(score)
    # sort
    inds = np.argsort(-np.array(scores))
    return [samples[i] for i in inds]


def process(method):
    log_file = 'vis_%s.txt' % method

    parsed = {}
    curr_key = None
    with open(log_file, 'r') as fs:
        for line in fs:
            raw = line.strip()
            if raw:
                quest_id, cont = _parse_line(raw)
                if quest_id is None:
                    parsed[curr_key].append(cont)
                else:
                    curr_key = quest_id
                    parsed[curr_key] = []
    # now let's sample the data
    target_id = 965492
    num_pick = 15
    sampled = random_pick(parsed[target_id], num_keep=num_pick)
    # sort according to scores
    sampled = sort_samples(sampled)
    # dump the results
    sv_file = 'result/vis_%s_rand%d.txt' % (method, num_pick)
    with open(sv_file, 'w') as fs:
        for line in sampled:
            fs.write('%s\n' % line)


if __name__ == '__main__':
    methods = ['deeperlstm', 'mlb2-att', 'n2mn', 'mlb-att']
    for method in methods:
        process(method)
