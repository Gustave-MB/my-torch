"""
NOTE: These test cases do not check the correctness of your solution,
      only whether anything has been implemented in functional.py.
      You are free to add your own test cases for checking correctness
"""

import numpy as np

from mytorch.nn.functional_hw2 import *


def test_conv1d_backward():
    bs = 3
    in_channel, out_channel = 5, 10
    input_size, kernel_size = 5, 3
    stride = 1
    output_size = ((input_size - kernel_size) // stride) + 1
    grad_output = np.zeros((bs, out_channel, output_size))
    a = np.zeros((bs, in_channel, input_size))
    b = np.zeros((out_channel, in_channel, kernel_size))
    c = np.zeros(out_channel)
    z = np.zeros((bs, in_channel, output_size))
    if conv1d_stride1_backward(grad_output, a, b, c):
        return True


def test_conv2d_backward():
    bs = 3
    in_channel, out_channel = 5, 10
    input_width, input_height = 5, 5
    kernel_size = 3
    stride = 1
    out_width = (input_width - kernel_size) // stride + 1
    out_height = (input_height - kernel_size) // stride + 1
    grad_output = np.zeros((bs, out_channel, out_width, out_height))
    a = np.zeros((bs, in_channel, input_width, input_height))
    b = np.zeros((out_channel, in_channel, kernel_size, kernel_size))
    c = np.zeros(out_channel)
    z = np.zeros((bs, in_channel, out_width, out_height))
    if conv2d_stride1_backward(grad_output, a, b, c):
        return True
