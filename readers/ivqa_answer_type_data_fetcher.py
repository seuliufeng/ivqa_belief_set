import numpy as np
import os
from util import load_hdf5, load_json, find_image_id_from_fname, save_hdf5
from multiprocessing import Process, Queue
from w2v_answer_encoder import MultiChoiceQuestionManger

if os.path.exists('/scratch/fl302/VQA/ResNet152'):
    FEAT_ROOT = '/scratch/fl302/VQA/ResNet152'
elif os.path.exists('/import/vision-ephemeral/fl302/data/VQA/ResNet152/resnet_res5c'):
    FEAT_ROOT = '/import/vision-ephemeral/fl302/data/VQA/ResNet152/resnet_res5c'
elif os.path.exists('data/resnet_res5c'):
    FEAT_ROOT = 'data/resnet_res5c'
else:
    print('Can''t find res_5c features')


class AttentionDataReader(object):
    def __init__(self, batch_size=32, subset='trainval',
                 output_feat=True, output_attr=True,
                 output_capt=True, output_qa=False,
                 output_ans_seq=False, attr_type='res152',
                 num_process=1, version='v1'):
        self._data_queue = None
        self._prefetch_procs = []
        self._batch_size = batch_size
        self._subset = subset
        self._output_im = output_feat
        self._output_attr = output_attr
        self._output_capt = output_capt
        self._output_qa = output_qa
        self._output_ans_seq = output_ans_seq
        self._attr_type = attr_type
        self._n_process = num_process
        self._version = version

    def start(self):
        self._data_queue = Queue(10)
        for proc_id in range(self._n_process):
            proc = AttentionDataPrefetcher(self._data_queue,
                                           proc_id,
                                           self._batch_size,
                                           self._subset,
                                           self._output_im,
                                           self._output_attr,
                                           self._output_capt,
                                           self._output_qa,
                                           self._output_ans_seq,
                                           self._attr_type,
                                           self._version)
            proc.start()
            self._prefetch_procs.append(proc)

        def cleanup():
            print 'Terminating BlobFetcher'
            for proc in self._prefetch_procs:
                proc.terminate()
                proc.join()

        import atexit
        atexit.register(cleanup)

    def stop(self):
        print 'Terminating BlobFetcher'
        for proc in self._prefetch_procs:
            proc.terminate()
            proc.join()

    def pop_batch(self):
        data = self._data_queue.get()
        return data


class AttentionDataPrefetcher(Process):
    def __init__(self, queue, proc_id, batch_size=32,
                 subset='trainval', output_im=True,
                 output_attr=True, output_capt=True,
                 output_qa=False, output_ans_seq=False,
                 attr_type='res152', version='v1'):
        super(AttentionDataPrefetcher, self).__init__()
        self._batch_size = batch_size
        self._proc_id = proc_id
        self._num_top_ans = 2000
        self._queue = queue
        self._subset = subset
        self._output_im = output_im
        self._output_attr = output_attr
        self._output_capt = output_capt
        self._output_qa = output_qa
        self._output_ans_seq = output_ans_seq
        self._attr_type = attr_type
        self._version_suffix = 'v2_' if version == 'v2' else ''
        self._dataset = 'VQA1.0' if version == 'v1' else 'VQA2.0'
        # data buffers
        self._images = None
        # vqa
        self._quest_len = None
        self._quest = None
        self._answer = None
        self._vqa_image_ids = None
        # captions
        self._captions = None
        self._caption_len = None
        self._image_id2capt_index = None
        self._n_capts_vqa_order = None
        # attributes
        self._attributes = None
        self._vqa_index2att_index = None
        # others
        self._num = None
        self._valid_ids = None
        self._load_data()

    def print_outputs(self):
        print('\n======== output statistic: ===========')
        print('Dataset version\t\t: %s' % self._dataset)
        print('Output Image\t\t: %r' % self._output_im)
        print('Output Attributes\t: %r' % self._output_attr)
        print('Output Caption\t\t: %r' % self._output_capt)
        print('Output QA pairs\t\t: %r' % self._output_qa)
        print('Output Answer Seq\t: %r' % self._output_ans_seq)
        print('======== output statistic: ===========\n')

    def _load_data(self):
        meta_file = 'data/%svqa_std_mscoco_%s.meta' % (self._version_suffix, self._subset)
        data_file = 'data/%svqa_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        d = load_json(meta_file)
        self._images = d['images']
        self._load_answer_type(d['quest_id'])
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['answer'].astype(np.int32)
        self._num = self._answer.size
        self._check_valid_answers()
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)
        # load caption data
        self._load_caption_data()
        # load attributes
        if self._attr_type == 'res152':
            self._load_global_image_feature()
        else:
            self._load_attributes()
        # load answer sequences
        self._load_answer_sequence()

    def _load_answer_type(self, quest_ids):
        answer_type_file = 'data/%sanswer_type_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        if not os.path.exists(answer_type_file):
            _mc_ctx = MultiChoiceQuestionManger(subset='train')
            answer_type_ids = [_mc_ctx.get_answer_type_coding(quest_id) for quest_id in quest_ids]
            answer_type_ids = np.array(answer_type_ids, dtype=np.int32)
            save_hdf5(answer_type_file, {'answer_type': answer_type_ids,
                                         'quest_ids': np.array(quest_ids, dtype=np.int32)})
        else:
            d = load_hdf5(answer_type_file)
            answer_type_ids = d['answer_type']
        self._answer_type = answer_type_ids

    def _load_answer_sequence(self):
        if not self._output_ans_seq:
            return
        data_file = 'data/%sanswer_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        d = load_hdf5(data_file)
        self._answer_seq = d['ans_arr']
        self._ans_seq_len = d['ans_len']

    def _load_global_image_feature(self):
        data_file = 'data/res152_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2att_index = np.array(vqa_index2att_index, dtype=np.int32)
        self._attributes = d['features']

    def _load_caption_data(self):
        if not self._output_capt:
            return
        # load data
        data_file = 'data/caption_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._captions = d['capt_arr']
        self._caption_len = d['capt_len']
        capt_image_ids = d['image_ids']
        image_id2capt_index = {}
        for i, image_id in enumerate(capt_image_ids):
            if image_id in image_id2capt_index:
                image_id2capt_index[image_id].append(i)
            else:
                image_id2capt_index[image_id] = [i]
        # length check
        n_capts_vqa_order = [len(image_id2capt_index[image_id])
                             for image_id in self._vqa_image_ids]
        self._n_capts_vqa_order = np.array(n_capts_vqa_order, dtype=np.int32)
        self._image_id2capt_index = image_id2capt_index

    def _load_attributes(self):
        data_file = 'data/attribute_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._attributes = d['att_arr'].astype(np.float32)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2att_index = np.array(vqa_index2att_index, dtype=np.int32)

    def _check_valid_answers(self):
        if self._output_qa and not self._output_ans_seq:
            self._valid_ids = np.where(self._answer < self._num_top_ans)[0]
        else:
            self._valid_ids = np.arange(self._num)

    def pop_batch(self):
        index = np.random.choice(self._valid_ids,
                                 size=self._batch_size,
                                 replace=False)
        outputs = []
        if self._output_im:
            feats = self._load_image_features(index)
            outputs.append(feats)
        if self._output_attr:
            attr = self._slice_attributes(index)
            outputs.append(attr)
        if self._output_capt:
            c, c_len = self._slice_caption(index)
            outputs += [c, c_len]
        if self._output_qa:
            q, q_len = self._slice_questions(index)
            a = self._slice_answers(index)
            # outputs += [q, q_len, a]
            outputs += [q, q_len]
        if self._output_ans_seq:
            ans_seq, ans_seq_len = self._slice_answer_sequence(index)
            outputs += [ans_seq, ans_seq_len]
        else:  # only one will happen
            outputs.append(self._answer_type[index])
        return outputs

    def get_next_batch(self):
        batch_data = self.pop_batch()
        self._queue.put(batch_data)

    def run(self):
        print('DataFetcher started')
        self.print_outputs()
        np.random.seed(self._proc_id)
        # very important, it ensures multiple processes don't generate the same index
        while True:
            self.get_next_batch()

    def _load_image_features(self, index):
        feats = []
        for idx in index:
            filename = self._images[idx]
            f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
            feats.append(f.transpose((1, 2, 0))[np.newaxis, ::])
        return np.concatenate(feats, axis=0).astype(np.float32)

    def _slice_questions(self, index):
        q_len = self._quest_len[index]
        max_len = q_len.max()
        seq = self._quest[index, :max_len]
        return seq, q_len

    def _slice_answer_sequence(self, index):
        a_len = self._ans_seq_len[index]
        max_len = a_len.max()
        seq = self._answer_seq[index, :max_len]
        return seq, a_len

    def _slice_answers(self, index):
        return self._answer[index]

    def _slice_caption(self, quest_index):
        image_ids = self._vqa_image_ids[quest_index]
        # sample captions
        n_capts = self._n_capts_vqa_order[quest_index]
        capt_tmp_inds = [np.random.randint(n) for n in n_capts]
        capt_index = [self._image_id2capt_index[image_id][capt_i]
                      for image_id, capt_i in zip(image_ids, capt_tmp_inds)]
        # slice captions
        capt_len = self._caption_len[capt_index]
        max_len = capt_len.max()
        capts = self._captions[capt_index, :max_len]
        return capts, capt_len.flatten()

    def _slice_attributes(self, index):
        attr_index = self._vqa_index2att_index[index]
        return self._attributes[attr_index, :]


class AttentionTestDataFetcher(object):
    def __init__(self, batch_size=32,
                 subset='trainval', output_feat=True,
                 output_attr=True, output_capt=True,
                 output_qa=False, output_ans_seq=False,
                 num_process=None, version='v1'):
        self._batch_size = batch_size
        self._num_top_ans = 2000
        self._subset = subset
        self._output_im = output_feat
        self._output_attr = output_attr
        self._output_capt = output_capt
        self._output_qa = output_qa
        self._output_ans_seq = output_ans_seq
        # data buffers
        self._images = None
        self._version_suffix = 'v2_' if version == 'v2' else ''
        # vqa
        self._quest_len = None
        self._quest = None
        self._answer = None
        self._vqa_image_ids = None
        # captions
        self._captions = None
        self._caption_len = None
        self._image_id2capt_index = None
        self._n_capts_vqa_order = None
        # attributes
        self._attributes = None
        self._vqa_index2att_index = None
        # others
        self._num = None
        self._valid_ids = None
        self._load_data()
        #
        self._idx = 0
        self._index = np.arange(self._num)
        self.print_outputs()

    def print_outputs(self):
        print('\n======== output statistic: ===========')
        # print('======== subset: %s ===========' % self._subset)
        print('Output Image\t\t: %r' % self._output_im)
        print('Output Attributes\t: %r' % self._output_attr)
        print('Output Caption\t\t: %r' % self._output_capt)
        print('Output QA pairs\t\t: %r' % self._output_qa)
        print('Output Answer Seq\t: %r' % self._output_ans_seq)
        print('======== output statistic: ===========\n')

    def _load_answer_type(self, quest_ids):
        answer_type_file = 'data/%sanswer_type_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        if not os.path.exists(answer_type_file):
            _mc_ctx = MultiChoiceQuestionManger(subset='val')
            answer_type_ids = [_mc_ctx.get_answer_type_coding(quest_id) for quest_id in quest_ids]
            answer_type_ids = np.array(answer_type_ids, dtype=np.int32)
            save_hdf5(answer_type_file, {'answer_type': answer_type_ids,
                                         'quest_ids': np.array(quest_ids, dtype=np.int32)})
        else:
            d = load_hdf5(answer_type_file)
            answer_type_ids = d['answer_type']
        self._answer_type = answer_type_ids

    def _load_data(self):
        meta_file = 'data/%svqa_std_mscoco_%s.meta' % (self._version_suffix, self._subset)
        data_file = 'data/%svqa_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        # load meta file
        d = load_json(meta_file)
        self._images = d['images']
        self._load_answer_type(d['quest_id'])
        self._quest_ids = np.array(d['quest_id'])
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)
        # load data file
        d = load_hdf5(data_file)
        self._quest = d['quest_arr'].astype(np.int32)
        self._quest_len = d['quest_len'].astype(np.int32)
        self._answer = d['answer'].astype(np.int32)
        self._num = self._answer.size
        self._check_valid_answers()
        vqa_image_ids = [find_image_id_from_fname(im_name) for im_name in self._images]
        self._vqa_image_ids = np.array(vqa_image_ids, dtype=np.int32)
        # load caption data
        self._load_caption_data()
        # load attributes
        self._load_attributes()
        # load answer sequences
        self._load_answer_sequence()
        # load answer type

    def _load_answer_sequence(self):
        if not self._output_ans_seq:
            return
        data_file = 'data/%sanswer_std_mscoco_%s.data' % (self._version_suffix, self._subset)
        d = load_hdf5(data_file)
        self._answer_seq = d['ans_arr']
        self._ans_seq_len = d['ans_len']

    def _load_caption_data(self):
        if not self._output_capt:
            return
        # load data
        data_file = 'data/caption_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._captions = d['capt_arr']
        self._caption_len = d['capt_len']
        capt_image_ids = d['image_ids']
        image_id2capt_index = {}
        for i, image_id in enumerate(capt_image_ids):
            if image_id in image_id2capt_index:
                image_id2capt_index[image_id].append(i)
            else:
                image_id2capt_index[image_id] = [i]
        # length check
        n_capts_vqa_order = [len(image_id2capt_index[image_id])
                             for image_id in self._vqa_image_ids]
        self._n_capts_vqa_order = np.array(n_capts_vqa_order, dtype=np.int32)
        self._image_id2capt_index = image_id2capt_index

    def _load_attributes(self):
        data_file = 'data/attribute_std_mscoco_%s.data' % self._subset
        d = load_hdf5(data_file)
        self._attributes = d['att_arr'].astype(np.float32)
        image_ids = d['image_ids']
        image_id2att_index = {image_id: i for i, image_id in enumerate(image_ids)}
        vqa_index2att_index = [image_id2att_index[image_id] for image_id in self._vqa_image_ids]
        self._vqa_index2att_index = np.array(vqa_index2att_index, dtype=np.int32)

    def _check_valid_answers(self):
        if self._output_qa and not self._output_ans_seq:
            self._valid_ids = np.where(self._answer < self._num_top_ans)[0]
        else:
            self._valid_ids = np.arange(self._num)

    def _get_sequencial_index(self):
        this_batch_size = min(self._batch_size, self._num-self._idx)
        index = self._index[self._idx:self._idx+this_batch_size]
        self._idx += this_batch_size
        return index

    def get_test_batch(self):
        index = self._get_sequencial_index()
        outputs = []
        if self._output_im:
            feats = self._load_image_features(index)
            outputs.append(feats)
        if self._output_attr:
            attr = self._slice_attributes(index)
            outputs.append(attr)
        if self._output_capt:
            c, c_len = self._slice_caption(index)
            outputs += [c, c_len]
        if self._output_qa:
            q, q_len = self._slice_questions(index)
            a = self._slice_answers(index)
            outputs += [q, q_len, a]
        if self._output_ans_seq:
            ans_seq, ans_seq_len = self._slice_answer_sequence(index)
            outputs += [ans_seq, ans_seq_len]
        else:  # only one will happen
            outputs.append(self._answer_type[index])
        quest_id = self._quest_ids[index]
        image_id = self._vqa_image_ids[index]
        outputs += [quest_id, image_id]
        return outputs

    def _load_image_features(self, index):
        feats = []
        for idx in index:
            filename = self._images[idx]
            f = np.load(os.path.join(FEAT_ROOT, filename + '.npz'))['x']
            feats.append(f.transpose((1, 2, 0))[np.newaxis, ::])
        return np.concatenate(feats, axis=0).astype(np.float32)

    def _slice_questions(self, index):
        q_len = self._quest_len[index]
        max_len = q_len.max()
        seq = self._quest[index, :max_len]
        return seq, q_len

    def _slice_answers(self, index):
        return self._answer[index]

    def _slice_answer_sequence(self, index):
        a_len = self._ans_seq_len[index]
        max_len = a_len.max()
        seq = self._answer_seq[index, :max_len]
        return seq, a_len

    def _slice_caption(self, quest_index):
        image_ids = self._vqa_image_ids[quest_index]
        # sample captions
        n_capts = self._n_capts_vqa_order[quest_index]
        capt_tmp_inds = [np.random.randint(n) for n in n_capts]
        capt_index = [self._image_id2capt_index[image_id][capt_i]
                      for image_id, capt_i in zip(image_ids, capt_tmp_inds)]
        # slice captions
        capt_len = self._caption_len[capt_index]
        max_len = capt_len.max()
        capts = self._captions[capt_index, :max_len]
        return capts, capt_len.flatten()

    def _slice_attributes(self, index):
        attr_index = self._vqa_index2att_index[index]
        return self._attributes[attr_index, :]

    @property
    def num_batches(self):
        from math import ceil
        n = ceil(self._num / float(self._batch_size))
        return int(n)

