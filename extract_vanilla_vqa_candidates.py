from __future__ import division
import tensorflow as tf
import numpy as np
import os
from util import update_progress, save_hdf5, save_json

from inference_utils.question_generator_util import SentenceGenerator
from vqa_config import ModelConfig
from models.vqa_base import BaseModel as model_fn
from readers.vqa_naive_data_fetcher import AttentionFetcher
from extract_vqa_word2vec_coding import SentenceEncoder
from w2v_answer_encoder import MultiChoiceQuestionManger

# _K = 10

# tf.flags.DEFINE_string("model_type", "VQA-Base",
#                        "Select a model to train.")
tf.flags.DEFINE_string("model_type", "VQA-BaseNorm",
                       "Select a model to train.")
tf.flags.DEFINE_string("version", "v1",
                       "Version of the dataset, v1 or v2.")
# tf.flags.DEFINE_string("checkpoint_dir", "model/%s_kpvqa_%s",
#                        "Model checkpoint file.")
tf.flags.DEFINE_string("checkpoint_dir", "model/%s_%s",
                       "Model checkpoint file.")
tf.flags.DEFINE_string("model_trainset", "kptrain",
                       "Which split is the model trained on")
tf.flags.DEFINE_string("subset", "kpval",
                       "Which split for feature extraction")
tf.flags.DEFINE_integer("num_cands", 5, "Num candidates")
tf.flags.DEFINE_bool("append_gt", False, "Num candidates")
tf.flags.DEFINE_string("result_format", "result/%s_vqa_OpenEnded_mscoco_%s2015_baseline_results.json",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
FLAGS = tf.flags.FLAGS
_K = FLAGS.num_cands

tf.logging.set_verbosity(tf.logging.INFO)


def extract_answer_proposals(checkpoint_path=None, subset='kpval'):
    batch_size = 100
    config = ModelConfig()
    # Get model function
    # model_fn = get_model_creation_fn(FLAGS.model_type)

    if FLAGS.append_gt:
        ann_set = 'train' if 'train' in subset else 'val'
        mc_ctx = MultiChoiceQuestionManger(subset=ann_set,
                                           load_ans=True,
                                           answer_coding='sequence')
    else:
        mc_ctx = None

    # build data reader
    reader = AttentionFetcher(batch_size=batch_size, subset=subset,
                              feat_type=config.feat_type, version=FLAGS.version)
    if checkpoint_path is None:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir % (FLAGS.version,
                                                                     FLAGS.model_type))
        checkpoint_path = ckpt.model_checkpoint_path
    print(checkpoint_path)

    # build and restore model
    model = model_fn(config, phase='test')
    # model.set_agent_ids([0])
    model.build()
    prob = model.prob

    sess = tf.Session(graph=tf.get_default_graph())
    tf.logging.info('Restore from model %s' % os.path.basename(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Create the vocabulary.
    top_ans_file = '../VQA-tensorflow/data/vqa_trainval_top2000_answers.txt'
    to_sentence = SentenceGenerator(trainset='trainval',
                                    top_ans_file=top_ans_file)
    w2v_encoder = SentenceEncoder()
    # to_sentence = SentenceGenerator(trainset='trainval')

    cands_meta = []
    cands_scores = []
    cands_coding = []
    quest_ids = []
    is_oov = []
    print('Running inference on split %s...' % subset)
    for i in range(reader.num_batches):
        if i % 10 == 0:
            update_progress(i / float(reader.num_batches))
        outputs = reader.get_test_batch()
        raw_ans = sess.run(
            prob, feed_dict=model.fill_feed_dict(outputs[:-2]))
        generated_ans = raw_ans.copy()
        generated_ans[:, -1] = -1.0  # by default do not predict UNK
        # print('Max: %0.3f, Min: %0.3f' % (raw_ans.max(), raw_ans.min()))

        gt_labels = outputs[-3]
        if FLAGS.append_gt:
            generated_ans[np.arange(gt_labels.size), gt_labels] = 10.0

        ans_cand_ids = np.argsort(-generated_ans, axis=1)

        q_ids = outputs[-2]

        if FLAGS.append_gt:
            assert (np.all(np.equal(ans_cand_ids[:, 0], gt_labels)))

        for quest_id, ids, cand_scs, _gt in zip(q_ids, ans_cand_ids,
                                                raw_ans, gt_labels):
            answers = []
            answer_w2v = []

            # check out of vocabulary
            is_oov.append(_gt == 2000)

            cands_scores.append(cand_scs[ids[:_K]][np.newaxis, :])
            for k in range(_K):
                aid = ids[k]
                if aid == 2000:  # gt is out of vocab
                    ans = mc_ctx.get_gt_answer(quest_id)
                else:
                    ans = to_sentence.index_to_top_answer(aid)
                answer_w2v.append(w2v_encoder.encode(ans))
                answers.append(ans)
            answer_w2v = np.concatenate(answer_w2v, axis=1)
            res_i = {'quest_id': int(quest_id), 'cands': answers}
            cands_meta.append(res_i)
            cands_coding.append(answer_w2v)
            quest_ids.append(quest_id)
    quest_ids = np.array(quest_ids, dtype=np.int32)
    is_oov = np.array(is_oov, dtype=np.bool)
    labels = np.zeros_like(quest_ids, dtype=np.int32)
    cands_scores = np.concatenate(cands_scores, axis=0).astype(np.float32)
    cands_coding = np.concatenate(cands_coding, axis=0).astype(np.float32)
    save_hdf5('data3/vqa_ap_w2v_coding_%s.data' % subset, {'cands_w2v': cands_coding,
                                                           'cands_scs': cands_scores,
                                                           'quest_ids': quest_ids,
                                                           'is_oov': is_oov,
                                                           'labels': labels})
    save_json('data3/vqa_ap_cands_%s.meta' % subset, cands_meta)
    print('\n\nExtraction Done!')
    print('OOV percentage: %0.2f' % (100.*is_oov.sum()/reader._num))


if __name__ == '__main__':
    extract_answer_proposals(subset=FLAGS.subset)
