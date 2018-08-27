import json
from w2v_answer_encoder import _tokenize_sentence
from collections import Counter


def compute_word_count(captions):
    assert (type(captions[0] == list))
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))
    return word_counts


def get_answers(subset):
    print('Loading annotation files [%s]...' % subset)
    ann_file = 'data/annotations/mscoco_%s2014_annotations.json' % subset
    d = json.load(open(ann_file, 'r'))
    ans_by_type, qid_by_type = {}, {}
    for info in d['annotations']:
        # quest_id = info['question_id']
        ans = [info['multiple_choice_answer']]
        qid = info['question_id']
        # ans = _tokenize_sentence(info['multiple_choice_answer'])
        ans_type = info['answer_type']
        if ans_type in ans_by_type:
            ans_by_type[ans_type].append(ans)
            qid_by_type[ans_type].append(qid)
        else:
            ans_by_type[ans_type] = [ans]
            qid_by_type[ans_type] = [qid]
    return ans_by_type, qid_by_type


def get_questions(subset):
    print('Loading question files [%s]...' % subset)
    quest_file = 'data/annotations/MultipleChoice_mscoco_%s2014_questions.json' % subset
    d = json.load(open(quest_file, 'r'))
    questions = d['questions']
    quest_vocab = {}
    print('Tokenize candidate answers...')
    for info in questions:
        question_id = info['question_id']
        quest = info['question']
        quest_vocab[question_id] = quest
    return quest_vocab


if __name__ == '__main__':
    ans_by_type, qid_by_type = get_answers('val')
    quests = get_questions('val')
    number_ans = ans_by_type['number']
    number_quest = qid_by_type['number']
    idx = 0
    keyword = 'no'
    ans_type = 'yes/no'
    for ans, qid in zip(ans_by_type[ans_type], qid_by_type[ans_type]):
        if keyword in ans[0]:
            idx += 1
            print('Q: %s\nA: %s\n' % (quests[qid], ans[0]))
    print('%d: %d' % (idx, len(quests)))
    ans_counts_by_type = {}
    # for t in ans_by_type:
    #     ans_counts_by_type[t] = compute_word_count(ans_by_type[t])

    ans_by_type = get_answers('val')
