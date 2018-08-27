try:
    import cv2
except:
    pass
import numpy as np
from scipy.io import loadmat


def _require_rgb(im):
    if len(im.shape) == 2:
        im = im[np.newaxis, ::]
        im = np.tile(im, [1, 1, 3])
    return im


def load_mean():
    d = loadmat('nets/resnet_mean.mat')
    return d['mean'].astype(np.float32)


def process_image(image_path, mean=None):
    # print(image_path)
    im = cv2.imread(image_path)
    im = _require_rgb(im)
    im = cv2.resize(im, (448, 448))
    im = im[:, :, [2, 1, 0]]  # convert RGB to BGR
    if mean is not None:
        im = im - mean
    else:
        im = im.astype(np.uint8)
    return im[np.newaxis, ::]

