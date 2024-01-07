import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        batch_size, _, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                Z[:, :, i, j] = np.tensordot(A[:, :, i:i+self.kernel_size, j:j+self.kernel_size],
                                             self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
        

        return Z
    
        
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        pad_dLdZ = np.pad(dLdZ, [(0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                                (self.kernel_size - 1, self.kernel_size - 1)], mode='constant')
        flipped_W = np.flip(self.W, axis=(2, 3))

        self.dLdb = np.sum(pad_dLdZ, axis=(0, 2, 3))

        batch_size, _, output_height, output_width = dLdZ.shape

        dLdA = np.zeros((batch_size, self.in_channels, self.A.shape[2], self.A.shape[3]))
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        for i in range(pad_dLdZ.shape[2] - self.kernel_size + 1):
            for j in range(pad_dLdZ.shape[3] - self.kernel_size + 1):
                if j< dLdZ.shape[3] and i< dLdZ.shape[2]:
                
                    self.dLdW += np.tensordot(dLdZ[:, :, i, j], self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size],
                                         axes=([0], [0]))
                
                dLdA[:, :, i, j] = np.tensordot(pad_dLdZ[:, :, i:i+self.kernel_size, j:j+self.kernel_size],
                                                 flipped_W, axes=([1, 2, 3], [0, 2, 3]))
                
        return dLdA 


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.padding = padding

        # Initialize Conv2d_stride1() and Downsample2d() instance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        if self.padding!=0:
             A = np.pad(A, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)])

        # Call conv2d_stride1
        conv_output = self.conv2d_stride1.forward(A)

        # Call downsample2d
        Z = self.downsample2d.forward(conv_output)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        downsample_grad = self.downsample2d.backward(dLdZ)

        # Call conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(downsample_grad)
        
        # Unpad the gradient
        if self.padding!=0:      
            dLdA = dLdA[:, :, self.padding:-self.padding, self.padding:-self.padding]


        return dLdA
