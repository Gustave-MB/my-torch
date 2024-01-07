import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = np.divide(1,(1+np.exp(-Z)))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (self.A - (self.A * self.A))

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        # self.A = np.divide(np.exp(Z) - np.exp(-Z), np.exp(Z) + np.exp(-Z))
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (1 - self.A**2)

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.Z = Z
        self.A = np.maximum(0, self.Z)
        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (self.Z > 0)

        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):

        self.Z = Z
        self.A = 0.5*self.Z*(1 + scipy.special.erf(self.Z/np.sqrt(2))) 

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (0.5*(1 + scipy.special.erf(self.Z/np.sqrt(2))) + (self.Z/np.sqrt(2*np.pi))*np.exp(-0.5*self.Z**2))

        return dLdZ  


    
class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        
        # Calculate the sum of exponentials for each row
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        
        # Compute the softmax probabilities
        self.A = exp_Z / sum_exp_Z
        
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = dLdA.shape[0] # TODO
        C = dLdA.shape[1] # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C)) # TODO



        # Initialize the final output dLdZ with all zeros.
        dLdZ = np.zeros((N, C))

        for i in range(N):
            for j in range(C):
                for k in range(C):
                    if j == k:
                        dLdZ[i, j] += self.A[i, k] * (1 - self.A[i, k]) * dLdA[i, k]
                    else:
                        dLdZ[i, j] += -self.A[i, j] * self.A[i, k] * dLdA[i, k]

        return dLdZ