import math
import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import torch.nn as nn
import torchvision

from torch.autograd import Function
from torch.nn import Parameter

class ConvSeluSVD(nn.Module):
	
	def __init__(self, inputSize, outputSize, stride=1, maxpool=False, ownBasis=False):
		super(ConvSeluSVD, self).__init__()

		self.inputSize = inputSize
		self.outputSize = outputSize
		self.stride = stride
		self.params = Parameter( torch.Tensor(outputSize * inputSize, 1,3).normal_(0, .02))
		self.selu = nn.SELU(True)
		self.bias = Parameter( torch.zeros(outputSize))
		self.maxpool = maxpool

		if ownBasis == True:

			self.basisWeights = Parameter( torch.Tensor(
				[[-0.21535662, -0.30022025, -0.26041868, -0.314888,  -0.45471892, -0.3971264,
					-0.26603645, -0.3896653,  -0.33079177],
					[ 0.34970352,  0.50572443,  0.36894855,  0.07661748,  0.08152138,  0.02740295,
					-0.28591475, -0.49375448, -0.38343033],
					[-0.3019736,  -0.02775075, 0.29349312, -0.50207216, -0.05312577,  0.5471206,
					-0.39858055, -0.09402011,  0.31616086]] ))

	def forward(self, input, basis_=None):
		
		if basis_ is None:
			basis_ = self.basisWeights

		basis = basis_.unsqueeze(0)
		
		basis = basis.expand(self.params.size(0), basis.size(1), basis.size(2) )

		weights = torch.bmm(self.params, basis )
		weights = weights.squeeze()
		
		weights = weights.view(self.outputSize, self.inputSize, 3,3)
		
		x = torch.nn.functional.conv2d(input, 
			weights, 
			bias=self.bias, 
			stride=self.stride, 
			padding=1, 
			dilation=1, 
			groups=1)

		x = self.selu(x)
		if self.maxpool:
			x = F.max_pool2d(x, 2)

		return x

class DialatedResUpscaleNNSELU(nn.Module):
	def __init__(self, inputSize, outputSize, depth=2, factor = 2,kernelSize = 3, dilations=[1, 2, 3], paddings=[1, 2, 3] , stride=1):
		super(DialatedResUpscaleNNSELU, self).__init__()

		self.upsample = nn.Upsample(scale_factor=factor, mode='nearest')
		self.node = ConvSeluSVD(inputSize, outputSize, ownBasis=True)
		
	def forward(self, input, basis=None):

		return self.upsample(self.node(input, basis))

class UNet(nn.Module):
	def __init__(self, opt):
		super(UNet, self).__init__()
		self.opt = opt

		self.encoderLevels = nn.ModuleList()
		self.encoderLevels.append(ConvSeluSVD(3,64,maxpool=True, ownBasis=True))
		self.encoderLevels.append(ConvSeluSVD(64,128,maxpool=True, ownBasis=True ))
		self.encoderLevels.append(ConvSeluSVD(128,256,maxpool=True, ownBasis=True ))
		self.encoderLevels.append(ConvSeluSVD(256,512,maxpool=True, ownBasis=True ))
		self.encoderLevels.append(ConvSeluSVD(512,1024,maxpool=True, ownBasis=True ))
		self.encoderLevels.append(ConvSeluSVD(1024, 1024,maxpool=True, ownBasis=True) )


		
		self.decoderLevels = nn.ModuleList()
		self.decoderLevels.append( DialatedResUpscaleNNSELU(128, 32)  )
		self.decoderLevels.append( DialatedResUpscaleNNSELU(256, 64)  )
		self.decoderLevels.append( DialatedResUpscaleNNSELU(512, 128)  )
		self.decoderLevels.append( DialatedResUpscaleNNSELU(768, 256)  )
		self.decoderLevels.append( DialatedResUpscaleNNSELU(1280, 256)  )
		self.decoderLevels.append( DialatedResUpscaleNNSELU(1024, 256)  )

		self.finalDecoder = ConvSeluSVD(32, 32, ownBasis=True) 
		self.collapse = nn.Conv2d(32, 3, kernel_size=1)

	#function to grab current flattened neural network weights
	def extract_gradients(self):
		tot_size = self.count_parameters()
		pvec = np.zeros(tot_size, np.float32)
		count = 0
		for param in self.parameters():
			sz = param.grad.data.cpu().numpy().flatten().shape[0]
			pvec[count:count + sz] = param.grad.data.cpu().numpy().flatten()
			count += sz
		return pvec.copy()
	
	#function to grab current flattened neural network weights
	def extract_parameters(self):
		tot_size = self.count_parameters()
		pvec = np.zeros(tot_size, np.float32)
		count = 0
		for param in self.parameters():
			sz = param.data.cpu().numpy().flatten().shape[0]
			pvec[count:count + sz] = param.data.cpu().numpy().flatten()
			count += sz
		return pvec.copy()

	#function to inject a flat vector of ANN parameters into the model's current neural network weights
	def inject_parameters(self, pvec):
		tot_size = self.count_parameters()
		count = 0

		for param in self.parameters():
			sz = param.data.cpu().numpy().flatten().shape[0]
			raw = pvec[count:count + sz]
			reshaped = raw.reshape(param.data.cpu().numpy().shape)
			param.data = torch.from_numpy(reshaped).cuda()
			count += sz

		return pvec

	#count how many parameters are in the model
	def count_parameters(self):
		count = 0
		for param in self.parameters():
			#print param.data.numpy().shape
			count += param.data.cpu().numpy().flatten().shape[0]
		return count



	def forward(self, input):
		x = input

		encoderOutputs = []
		for e in self.encoderLevels:
		#for e in self.encoderLevels:
			x = e(x)
			encoderOutputs.append(x)
		
		previousDecoderOutput = None
		for i in range(len(self.decoderLevels)):

			#print(i)
			idx =-(i+1)
			d = self.decoderLevels[idx]

			encoderOutput = encoderOutputs[idx]
			#print('{} {}'.format(i, encoderOutput.size()))
			decoderInput = encoderOutput
			if previousDecoderOutput is not None:
				#print(previousDecoderOutput.size())
				decoderInput = torch.cat([encoderOutput, previousDecoderOutput], dim=1)

			#print(decoderInput.size())
			previousDecoderOutput = d(decoderInput ) 
			#print(previousDecoderOutput.size())

			#print(previousDecoderOutput.size())
		x = self.finalDecoder (previousDecoderOutput )
		output =  F.tanh(self.collapse(x))
		
		return output
