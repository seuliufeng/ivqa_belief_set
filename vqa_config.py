from __future__ import division


class ModelConfig(object):
    def __init__(self):
        # self.im_dim = 4096
        # self.im_dim = 1000
        self.im_dim = 2048
        self.word_embed_dim = 512
        self.decoder_size = 512
        # self.vocab_size = 12605
        # self.vocab_size = 10000
        # self.vocab_size = 16000
        self.vocab_size = 15954
        self.im_embed_dim = 1024
        self.quest_embed_dim = 1024
        self.joint_embed_dim = 1024
        self.answer_embed_dim = 128

        self.ans_vocab_size = 2001
        # don't compute loss wrt out vocabulary samples
        self.disable_ans_id = 2000

        self.num_rnn_cells = 512
        self.keep_prob = 0.5
        self.quest_max_len = 23
        self.num_preprocess_threads = 1

        self.batch_size = 32

        # self.image_feature_key = 'image/image_resnet152'
        self.image_feature_key = 'image/image_resnet152att'
        # self.image_feature_key = 'image/encoded'
        # self.feat_type = 'Inception'
        self.feat_type = 'Res5c'
        # self.feat_type = 'Mix7c'
        if self.feat_type == 'Res5c' or self.feat_type == 'Mix7c':
            self.n_lat = 14
            self.im_chn = 2048
        else:
            self.n_lat = 8
            self.im_chn = 1000
        # self.embed_dim = 512
        self.embed_dim = 1200

        # self.n_experts = 4
        self.n_experts = 4
        self.use_sum_op = False

        self.n_glimpses = 1
        self.train_resnet = False
        self.use_feed = True
        self.resnet_checkpoint_file = 'nets/resnet_v1_152.ckpt'
        # self.image_feature_key = 'image/image_incept_prob'
        self.encoder = 'gru'  # choose from {'skip-thought' and 'gru'}
        # self.image_feature_key = 'image/image_vgg19'


class TrainConfig(object):
    def __init__(self):
        # Train Parameters setting
        # starter_learning_rate = 3e-4
        self.learning_rate_decay_start = -1  # at what iteration to start decaying learning rate? (-1 = dont)
        self.batch_size = 16  # batch_size for each iterations
        self.decay_factor = 0.3333333333333333333333
        self.decay_step = 60000
        # self.decay_factor = 0.99997592083
        self.checkpoint_path = 'model_save/'
        self.max_itr = 200000
        self.clip_gradients = 10.0
        self.initial_learning_rate = 0.0002
        # self.initial_learning_rate = 0.00003
        # self.initial_learning_rate = 0.0001
        self.max_checkpoints_to_keep = 3
        self.optimizer = 'Adam'
        # self.optimizer = 'SGD'


class CaptionTrainConfig(object):
    def __init__(self):
        # Train Parameters setting
        # starter_learning_rate = 3e-4
        self.learning_rate_decay_start = -1  # at what iteration to start decaying learning rate? (-1 = dont)
        self.batch_size = 16  # batch_size for each iterations
        self.decay_factor = 0.3333333333333333333333
        self.decay_step = 60000
        # self.decay_factor = 0.99997592083
        self.checkpoint_path = 'model_save/'
        self.max_itr = 150000
        self.clip_gradients = 10.0
        self.initial_learning_rate = 0.0005
        # self.initial_learning_rate = 0.00003
        # self.initial_learning_rate = 0.0001
        self.max_checkpoints_to_keep = 3
        self.optimizer = 'Adam'
        # self.optimizer = 'SGD'


class TestConfig(object):
    def __init__(self):
        self.batch_size = 100
        self.result_dir = 'result'

    def get_result_file(self, model_path):
        import os
        model_name = os.path.basename(model_path)
        return os.path.join(self.result_dir,
                            'OpenEnded_%s_results.json' % model_name)


class QuestionGeneratorConfig(ModelConfig):
    def __init__(self):
        ModelConfig.__init__(self)
        self.im_dim = 1000
        self.num_decoder_lstm_cells = 512
        self.word2vec_dim = 300
        self.answer_embed_dim = self.im_embed_dim
        self.keep_prob = 0.5
        self.word_embed_dim = 512
        self.vocab_size = 10000
        self.batch_size = 500
        self.image_feature_key = 'image/image_incept_prob'
        # parameters for lstm answer encoder
        self.num_ans_enc_cells = 512
        self.ans_vocab_size = 6900
        self.pred_answer = True
        self.num_valid_answer = 1000


class QuestionGeneratorTrainConfig(TrainConfig):
    def __init__(self):
        TrainConfig.__init__(self)
        # self.clip_gradients = 10.0
        # self.optimizer = 'Adam'
        # self.initial_learning_rate = 2.0
        # self.learning_rate_decay_factor = 0.5
        # self.num_examples_per_epoch = 586363  # not correct
        # self.num_epochs_per_decay = 8.0


class DatasetConfig(object):
    def __init__(self):
        self.json_file = 'data/data_prepro.json'
        self.img_file = 'data/data_img.h5'
        self.quest_file = 'data/data_prepro.h5'
        self.apply_norm = True
