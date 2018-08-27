from util import *


def evaluate_question(result_file):
    from eval_vqa_question import QuestionEvaluator
    from util import get_dataset_root
    vqa_data_root, _ = get_dataset_root()
    base_name = os.path.splitext(result_file)
    res_mat_file = '%s_lig.mat' % base_name[0]

    subset = 'val'
    annotation_file = '%s/Annotations/mscoco_%s2014_annotations.json' % (vqa_data_root, subset)
    question_file = '%s/Questions/OpenEnded_mscoco_%s2014_questions.json' % (vqa_data_root, subset)

    evaluator = QuestionEvaluator(annotation_file, question_file)
    evaluator.evaluate(result_file)
    evaluator.save_results(eval_res_file=res_mat_file)
    return evaluator.get_overall_cider()


def load_and_evaluate(model_type):
    res_file = 'result/test/quest_vaq_%s_kptest.json' % (model_type.upper())
    evaluate_question(res_file)


if __name__ == '__main__':
    evaluate_question('result/test/beamsearch_vaq_VAQ-LSTM-DEC-SUP_kptest.json')
    exit(0)
    for model_type in ['VAQ-IAS', 'VAQ-lstm', 'VQG', 'VAQ-A']:
        load_and_evaluate(model_type)
