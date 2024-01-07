import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.max(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)

        _, _, output_height, output_width = dLdZ.shape

        for i in range(output_height):
            for j in range(output_width):
                mask = (self.A[:, :, i:i+self.kernel, j:j+self.kernel] == np.max(self.A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3), keepdims=True))
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += mask * (dLdZ[:, :, i, j])[:, :, None, None]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        input_width = self.kernel + output_width - 1
        input_height = self.kernel + output_height - 1

        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, None, None]

        # Divide by kernel size to account for mean
        dLdA /= self.kernel * self.kernel

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        conv_output = self.maxpool2d_stride1.forward(A) 
        Z = self.downsample2d.forward(conv_output)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsample_grad = self.downsample2d.backward(dLdZ)   

        dLdA = self.maxpool2d_stride1.backward(downsample_grad)  

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        conv_output = self.meanpool2d_stride1.forward(A) 
        Z = self.downsample2d.forward(conv_output)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        downsample_grad = self.downsample2d.backward(dLdZ)   

        dLdA = self.meanpool2d_stride1.backward(downsample_grad)  

        return dLdA
