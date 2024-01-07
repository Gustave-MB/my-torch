import numpy as np
from mytorch.nn.functional_hw1 import *
from mytorch.nn.functional_hw2 import *

class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        raise NotImplementedError

    def backward(self, dLdZ):
        raise NotImplementedError


class Downsample2d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        raise NotImplementedError


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        raise NotImplementedError

    def backward(self, dLdZ):
        raise NotImplementedError


class Downsample1d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        raise NotImplementedError
