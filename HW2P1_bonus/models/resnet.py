import sys
sys.path.append('mytorch')

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		self.layers = [Conv2d(in_channels, out_channels, kernel_size, stride, padding),
				 BatchNorm2d(out_channels), ReLU(),
				 Conv2d(out_channels, out_channels, 1, 1, 0), BatchNorm2d(out_channels)] 							#TODO					

	def forward(self, A):
		self.Z = A
		for layer in self.layers:
			self.Z = layer.forward(self.Z )
		return self.Z 

	def backward(self, grad): 
		self.dLdA = grad
		for layer in reversed(self.layers):
			self.dLdA = layer.backward(self.dLdA )
		return self.dLdA


class ResBlock(object):
	def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
		
		self.convolution_layers = ConvBlock( in_channels, out_channels, filter_size, stride, padding) 
						
		self.final_activation = ReLU()		 

		if stride != 1 or in_channels != out_channels or filter_size!=1 or padding!=0:
			self.residual_connection = [Conv2d(in_channels, out_channels, filter_size, stride, padding),
							   BatchNorm2d(out_channels)] 		#TODO
		else:
			self.residual_connection = [Identity()]			#TODO 


	def forward(self, A):
		Z = A
		'''
		Implement the forward for convolution layer.

		'''
		Z = self.convolution_layers.forward(Z) 
			

		'''
		Add the residual connection to the output of the convolution layers

		'''
		res = A
		for connection in self.residual_connection:
			res = connection.forward(res) 

		Z = np.add(Z, res) 
		

		'''
		Pass the the sum of the residual layer and convolution layer to the final activation function
		'''
		Z = self.final_activation.forward(Z) 

		return Z
	

	def backward(self, grad):

		'''
		Implement the backward of the final activation
		'''
		grad = self.final_activation.backward(grad) 


		'''
		Implement the backward of residual layer to get "residual_grad"
		'''
		residual_grad = grad
		for connection in reversed(self.residual_connection):
			residual_grad = connection.backward(residual_grad) 

		


		'''
		Implement the backward of the convolution layer to get "convlayers_grad"
		'''
		convlayers_grad = self.convolution_layers.backward(grad)  


		'''
		Add convlayers_grad and residual_grad to get the final gradient 
		'''
		gradient_final = np.add(convlayers_grad, residual_grad)  



		return gradient_final
