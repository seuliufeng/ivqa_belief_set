import sys
from util import get_dataset_root

dataDir, _ = get_dataset_root()
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' % (dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools/' % (dataDir))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os


def write_result_log(model_file, model_type,
                     total_res, per_type_res):
    fname = 'result/detail_%s.txt' % model_type
    with open(fname, 'a+') as f:
        f.write('%s\n' % model_file)
        for type in per_type_res:
            f.write('%s\t: %0.2f\n' % (type, per_type_res[type]))
        f.write('overall\t: %0.2f\n\n' % total_res)


def evaluate_model(resFile, quest_ids, subset='val', version='v1'):
    ans_type = None
    # set up file names and paths
    taskType = 'OpenEnded'
    dataType = 'mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
    dataSubType = '%s2014' % subset
    if version == 'v1':
        annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
        quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
    elif version == 'v2':
        anno_dir = '/import/vision-ephemeral/fl302/data/VQA2.0'
        annFile = '%s/v2_%s_%s_annotations.json' % (anno_dir, dataType, dataSubType)
        quesFile = '%s/v2_%s_%s_%s_questions.json' % (anno_dir, taskType, dataType, dataSubType)
    else:
        raise Exception('unknown version, v1 or v2')
    imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
    resultType = 'fake'
    fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

    # An example result json file has been provided in './Results' folder.

    [accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
        '%s/Results/%s_%s_%s_%s_%s.json' % (dataDir, taskType, dataType, dataSubType, \
                                            resultType, fileType) for fileType in fileTypes]

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate(quesIds=quest_ids)

    # print accuracies
    print "\n"
    print "Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall'])
    print "Per Question Type Accuracy is the following:"
    for quesType in vqaEval.accuracy['perQuestionType']:
        print "%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType])
    print "\n"
    print "Per Answer Type Accuracy is the following:"
    for ansType in vqaEval.accuracy['perAnswerType']:
        print "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType])
    print "\n"

    if ans_type is None:
        return vqaEval.accuracy['overall'], vqaEval.accuracy['perAnswerType']
    else:
        return vqaEval.accuracy['overall'], vqaEval.accuracy['perAnswerType'][ans_type]


if __name__ == '__main__':
    from config import TestConfig

    config = TestConfig()
    res_file = config.get_result_file('model-75001')
    evaluate_model('OpenEnded_mscoco_lstm_results.json')
