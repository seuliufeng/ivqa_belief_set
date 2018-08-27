
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inference_utils import inference_wrapper_base
from vqa_model_creater import get_model_creation_fn



class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with a ShowAndTellModel."""

    def __init__(self):
        super(InferenceWrapper, self).__init__()

    def build_model(self, model_config):
        model_creator = get_model_creation_fn(model_config.model_type)
        model = model_creator(model_config, phase=model_config.phase)
        model.build()
        return model

    def feed_image(self, sess, inputs):
        image, ans = inputs
        initial_state = sess.run(fetches="vaq/initial_state:0",
                                 feed_dict={"image_feed:0": image,
                                            "ans_feed:0": ans})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        softmax_output, state_output = sess.run(
            fetches=["vaq/softmax:0", "vaq/state:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "vaq/state_feed:0": state_feed,
            })
        return softmax_output, state_output, None
