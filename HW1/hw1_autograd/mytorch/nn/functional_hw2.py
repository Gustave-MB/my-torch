import numpy as np
from mytorch.autograd_engine import Autograd


# NOTE: backward functions below are for hw2-autograd only
def conv1d_stride1_backward(dLdZ, A, W, bias):
    raise NotImplementedError


def conv2d_stride1_backward(dLdZ, A, W, bias):
    raise NotImplementedError


def downsampling1d_backward(dLdZ, A, downsampling_factor):
    raise NotImplementedError


def downsampling2d_backward(dLdZ, A, downsampling_factor):
    raise NotImplementedError


def flatten_backward(dLdZ, A):
    raise NotImplementedError
