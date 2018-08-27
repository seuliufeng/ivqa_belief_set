import tensorflow as tf
from models.model_creater import get_model_creation_fn



model_fn = get_model_creation_fn('LM')
model = model_fn()
model.build()

sess = tf.Session()
model.set_session(sess)
model.setup_model()

import numpy as np
batch_size = 32
fake = np.random.randint(low=0, high=1000, size=(batch_size, 20)).astype(np.int32)
fake_len = np.ones(shape=(batch_size,), dtype=np.int32)
prob = model.inference([fake, fake_len])
print(prob)
loss = model.trainstep([fake, fake_len, fake, fake_len])
print(loss)