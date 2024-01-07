import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        skip = self.upsampling_factor

        Z = np.zeros((batch_size, in_channels, output_width))
        Z[:, :, ::skip] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        skip = self.upsampling_factor

        dLdA = dLdZ[:, :, ::skip]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """        

        self.input_width = A.shape[-1]
        skip = self.downsampling_factor

        Z = A[:, :, ::skip]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        skip = self.downsampling_factor

        dLdA = np.zeros((batch_size, in_channels, self.input_width))
        dLdA[:, :, ::skip] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size, in_channels, input_height, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        output_height = self.upsampling_factor * (input_height - 1) + 1
        skip = self.upsampling_factor

        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        Z[:, :, ::skip, ::skip] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        skip = self.upsampling_factor
        dLdA = dLdZ[:, :, ::skip, ::skip]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.input_width = A.shape[-1]
        self.input_height = A.shape[-2]
        skip = self.downsampling_factor

        Z = A[:, :, ::skip, ::skip]

        return Z
    

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        skip = self.downsampling_factor

        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))
        dLdA[:, :, ::skip, ::skip] = dLdZ

        return dLdA
