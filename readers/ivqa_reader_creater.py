# VAQ
from readers import ivqa_general_data_fetcher
from readers import ivqa_answer_type_data_fetcher
from readers import ivqa_full_data_fetcher
from readers import ivqa_rerank_data_fetcher
from readers import vqa_general_data_fetcher
from readers import vqa_general_data_fetcher_v7w
from readers import vqg_general_data_fetcher
from readers import ivqa_rl_data_fetcher
from readers import ivqa_basicplus_data_fetcher
from readers import vert_base_data_fetcher

_NUM_PREFETCH_PROCESS = 2


def _reader_config():
    reader_config = {
        'VQG': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                'use_res5c': False, 'use_attr': True, 'output_answer': False,
                'attr_type': 'res152'},
        'VAQ-AT': {'train_fetcher': ivqa_answer_type_data_fetcher.AttentionDataReader,
                   'test_fetcher': ivqa_answer_type_data_fetcher.AttentionTestDataFetcher,
                   'use_res5c': False, 'use_attr': True, 'output_answer': False,
                   'attr_type': 'res152'},
        'VAQ-A': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                  'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                  'use_res5c': False, 'use_attr': False, 'output_answer': True,
                  'attr_type': 'res152'},
        'iVQA-BasicPlus': {'train_fetcher': ivqa_basicplus_data_fetcher.AttentionDataReader,
                           'test_fetcher': ivqa_basicplus_data_fetcher.AttentionTestDataFetcher,
                           'use_res5c': False, 'use_attr': True, 'output_answer': True,
                           'attr_type': 'res152'},
        'VAQ-Att': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': True, 'use_attr': False, 'output_answer': True,
                    'attr_type': 'semantic'},
        'VAQ-Var': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': False, 'use_attr': True, 'output_answer': True,
                    'attr_type': 'res152'},
        'VAQ-VarDS': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                      'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                      'use_res5c': False, 'use_attr': True, 'output_answer': True,
                      'attr_type': 'semantic'},
        'VAQ-Epoch': {'train_fetcher': ivqa_rl_data_fetcher.AttentionDataReader,
                      'test_fetcher': ivqa_rl_data_fetcher.AttentionTestDataFetcher,
                      'use_res5c': False, 'use_attr': True, 'output_answer': True,
                      'attr_type': 'res152'},
        'VAQ-EpochAtt': {'train_fetcher': ivqa_rl_data_fetcher.AttentionDataReader,
                         'test_fetcher': ivqa_rl_data_fetcher.AttentionTestDataFetcher,
                         'use_res5c': True, 'use_attr': True, 'output_answer': True,
                         'attr_type': 'res152'},
        'VQG-Var': {'train_fetcher': vqg_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': vqg_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': False, 'use_attr': True, 'output_answer': True,
                    'attr_type': 'res152'},
        'VQA-Var': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': False, 'use_attr': True, 'output_answer': True,
                    'attr_type': 'res152'},
        'VQA-VarDS': {'train_fetcher': vqa_general_data_fetcher.AttentionDataReader,
                      'test_fetcher': vqa_general_data_fetcher.AttentionTestDataFetcher,
                      'use_res5c': False, 'use_attr': True, 'output_answer': True,
                      'attr_type': 'res152'},
        'V7W-VarDS': {'train_fetcher': vqa_general_data_fetcher_v7w.AttentionDataReader,
                      'test_fetcher': vqa_general_data_fetcher_v7w.AttentionTestDataFetcher,
                      'use_res5c': False, 'use_attr': True, 'output_answer': True,
                      'attr_type': 'res152'},
        'Fusion': {'train_fetcher': ivqa_rerank_data_fetcher.AttentionDataReader,
                   'test_fetcher': ivqa_rerank_data_fetcher.AttentionTestDataFetcher,
                   'use_res5c': False, 'use_attr': True, 'output_answer': False,
                   'attr_type': 'res152'},
        'Fusionv1': {'train_fetcher': ivqa_rerank_data_fetcher.AttentionDataReader,
                     'test_fetcher': ivqa_rerank_data_fetcher.AttentionTestDataFetcher,
                     'use_res5c': False, 'use_attr': True, 'output_answer': False,
                     'attr_type': 'res152'},
        'VAQ-SAT': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': True, 'use_attr': False, 'output_answer': True,
                    'attr_type': 'semantic'},
        'VAQ-2Att': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                     'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                     'use_res5c': True, 'use_attr': True, 'output_answer': True,
                     'attr_type': 'semantic'},
        'VAQ-VIS': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                    'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                    'use_res5c': True, 'use_attr': True, 'output_answer': True,
                    'attr_type': 'res152'},
        'VAQ-VVIS': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                     'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                     'use_res5c': False, 'use_attr': True, 'output_answer': True,
                     'attr_type': 'res152'},
        'VAQ-RL': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                   'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                   'use_res5c': True, 'use_attr': True, 'output_answer': True,
                   'attr_type': 'semantic'},
        'VAQ-Mixer': {'train_fetcher': ivqa_general_data_fetcher.AttentionDataReader,
                      'test_fetcher': ivqa_general_data_fetcher.AttentionTestDataFetcher,
                      'use_res5c': True, 'use_attr': True, 'output_answer': True,
                      'attr_type': 'semantic'},
        'VAQ-CA': {'train_fetcher': ivqa_full_data_fetcher.AttentionDataReader,
                   'test_fetcher': ivqa_full_data_fetcher.AttentionTestDataFetcher,
                   'use_res5c': True, 'use_attr': True, 'output_answer': True,
                   'attr_type': 'semantic'}
    }
    reader_config['VQG-Att'] = reader_config['VAQ-2Att']
    reader_config['iVQA-Basic'] = reader_config['VAQ-Var']
    reader_config['VQA-VarDS1'] = reader_config['VQA-VarDS']
    reader_config['VQA-VarSR'] = reader_config['VQA-VarDS']
    reader_config['VQA-VarSRSG'] = reader_config['VQA-VarDS']
    reader_config['VQA-VarSRE2E'] = reader_config['VQA-VarDS']
    reader_config['VAQ-VarDST'] = reader_config['VAQ-VarDS']
    reader_config['VAQ-VarDSA'] = reader_config['VAQ-VarDS']
    reader_config['VAQ-VarDSA-Q'] = reader_config['VAQ-VarDSA']
    reader_config['VAQ-VarDSA-IAQ'] = reader_config['VAQ-VarDSA']
    reader_config['VAQ-VarDSA3'] = reader_config['VAQ-VarDS']
    reader_config['VAQ-VarDSDC'] = reader_config['VAQ-VarDS']
    return reader_config


def create_reader(model_type, phase='train'):
    config = _reader_config()
    model_config = config[model_type]
    create_fn = model_config['train_fetcher'] if phase == 'train' else model_config['test_fetcher']
    output_attr = model_config['use_attr']
    output_res5c = model_config['use_res5c']
    output_answer = model_config['output_answer']
    attr_type = model_config['attr_type']
    print(model_type)
    print(attr_type)
    output_qa = phase == 'train' or model_type == 'VAQ-CA' or model_type == 'VAQ-Epoch' or model_type == 'VAQ-VIS' \
                or model_type == 'VAQ-EpochAtt' or model_type == 'VAQ-VVIS'

    # output_qa = phase == 'train' or model_type == 'VAQ-CA' or 'Fusion' or 'VQA-Var'\
    #             or 'VQA-VarDS' or 'V7W' in model_type  # or model_type == 'VAQ-SAT'

    def _creater_general_reader(batch_size, subset='kptrain', version='v1'):
        reader = create_fn(batch_size,
                           subset,
                           output_feat=output_res5c,
                           output_attr=output_attr,
                           output_capt=False,
                           output_qa=output_qa,
                           output_ans_seq=output_answer,
                           attr_type=attr_type,
                           num_process=_NUM_PREFETCH_PROCESS,
                           version=version)
        return reader

    return _creater_general_reader
