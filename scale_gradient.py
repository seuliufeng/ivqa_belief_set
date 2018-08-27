import tensorflow as tf
from tensorflow.python.framework import ops


class ScaleGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, ratio=1.0):
        grad_name = "ScaleGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _scale_gradients(op, grad):
            return [grad * ratio]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


scale_gradient = ScaleGradientBuilder()


def debug_scale_gradient():
    import numpy as np
    import numpy.random as nr
    ratio = -0.1
    x = tf.constant(nr.rand(4))
    y = scale_gradient(x, ratio=ratio)
    loss = tf.nn.l2_loss(y)
    grad_y = tf.gradients(loss, y)[0]
    grad_x = tf.gradients(loss, x)[0]
    sess = tf.InteractiveSession()
    print('x:')
    print(x.eval())
    v_grad_y = grad_y.eval()
    print('\ngrad_y:')
    print(v_grad_y)
    print('\ngrad_x')
    v_grad_x = grad_x.eval()
    print(v_grad_x)
    print(v_grad_x == v_grad_y*ratio)


if __name__ == '__main__':
    debug_scale_gradient()