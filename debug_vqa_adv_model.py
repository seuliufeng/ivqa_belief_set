import tensorflow as tf
# from models.model_creater import get_model_creation_fn
from models.vqa_adversary import BaseModel


# model_fn = get_model_creation_fn('LM')
from vqa_config import ModelConfig

model = BaseModel(ModelConfig(), phase='train')
model.build()

sess = tf.Session()
model.set_session(sess)
model.setup_model()

import numpy as np
batch_size = 8
quest = np.random.randint(low=0, high=1000, size=(batch_size, 20)).astype(np.int32)
quest_len = np.ones(shape=(batch_size,), dtype=np.int32)
images = np.random.rand(batch_size, 2048)
labels = np.random.randint(low=0, high=2000, size=(batch_size,), dtype=np.int32)
prob = model.inference([images, quest, quest_len])
print(prob)
loss = model.trainstep([images, quest, quest_len, labels])
print(loss)
