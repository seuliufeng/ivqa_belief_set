from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from inference_utils import inference_wrapper_base
from models.model_creater import get_model_creation_fn


class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with a ShowAndTellModel."""

    def __init__(self):
        self.image = None
        self.answer_feed = None
        self.answer_len_feed = None
        super(InferenceWrapper, self).__init__()

    def build_model(self, model_config):
        model_creator = get_model_creation_fn(model_config.model_type)
        model = model_creator(model_config, phase=model_config.phase)
        model.build()
        return model

    def feed_image(self, sess, inputs):
        image, attr, ans_seq, ans_len = inputs
        image = np.squeeze(image)
        attr = attr.flatten()
        ans_seq = ans_seq.flatten()
        ans_len = ans_len.flatten()
        self.image = image
        self.answer_feed = ans_seq
        self.answer_len_feed = ans_len
        initial_state = sess.run(fetches="inverse_vqa/initial_state:0",
                                 feed_dict={"image_feed:0": image,
                                            "attr_feed:0": attr,
                                            "ans_feed:0": ans_seq,
                                            "ans_len_feed:0": ans_len})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        softmax_output, state_output = sess.run(
            fetches=["inverse_vqa/softmax:0", "inverse_vqa/state:0"],
            feed_dict={
                "image_feed:0": self.image,
                "ans_feed:0": self.answer_feed,
                "ans_len_feed:0": self.answer_len_feed,
                "input_feed:0": input_feed,
                "inverse_vqa/state_feed:0": state_feed,
            })
        return softmax_output, state_output, None
