import torch
import dataloader
import argparse
import torchvision
import model
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import gc
import itertools
import time

import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.morphology import watershed
import torchvision.transforms as transforms

import sklearn
import sklearn.decomposition
import random
import torchvision.models as models
import cv2
import multiprocessing

#from apex import amp
#amp_handle = amp.init()

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataroota', default=[
	'/home/lolz0r/domain/data/billted/',
	'/home/lolz0r/domain/data/lifeofpi/',
	'/home/lolz0r/coco2017/train2017/'
	], type=str)

parser.add_argument('--nthread', default=3, type=int)

parser.add_argument('--loadSize', default=600, type=int)
parser.add_argument('--imgSize', default=512, type=int)
parser.add_argument('--batchSize', default=12, type=int)
parser.add_argument('--itersPerEpoch', default=10000, type=int)

parser.add_argument('--encoderStep', default=64, type=int)
parser.add_argument('--levels', default=1, type=int)

opt = parser.parse_args()

trainDataset = torch.utils.data.DataLoader(dataloader.ABDataloader(opt), 
	shuffle=True, batch_size=opt.batchSize, num_workers=opt.nthread, pin_memory=True)
autoEncoder =  model.UNet(opt).cuda()
netGStateDict = autoEncoder.state_dict()

loadedSD = torch.load('./saves/autoEncoder--3.983832822715064.pth')
for k in netGStateDict.keys():
	print(k)
	if k in loadedSD and netGStateDict[k].size() == loadedSD[k].size():
		netGStateDict[k] = loadedSD[k]
		print('... copied')
autoEncoder.load_state_dict(netGStateDict  )

autoEncoder.cuda()

#autoEncoder.load_state_dict(torch.load('/home/lolz0r/domain/linesToImg/saves/autoEncoder-0.12010071901894874.pth'))
#autoEncoder.load_state_dict(torch.load('/home/lolz0r/domain/linesToImg/saves/autoEncoder-0.8203608310765785.pth'))
#autoEncoder =  torch.nn.DataParallel(autoEncoder, device_ids=[0, 1])

#optimizerEnc = YFOptimizer(encoder.parameters())
#optimizer = YFOptimizer(autoEncoder.parameters())
#optimizerDecB = YFOptimizer(decoderB.parameters())
 
#optimizer = torch.optim.SGD(autoEncoder.parameters(), lr=0.0005, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=0.001, amsgrad=True )

itercount = 0
evalFirst = False


def pearsonr(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	r_num = xm.dot(ym)
	r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
	r_val = r_num / r_den
	return r_val

class NetPerLayer(nn.Module):
	def __init__(self):
		super(NetPerLayer, self).__init__()

		net = torchvision.models.resnet18(pretrained=True)
		net.eval()

		self.nodeLevels = nn.ModuleList()
		self.nodeLevels.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
		self.nodeLevels.append(net.layer1)
		self.nodeLevels.append(net.layer2)
		self.nodeLevels.append(net.layer3)
		self.nodeLevels.append(net.layer4)
		#self.nodeLevels.append(nn.MaxPool2d(8))

		
	def forward(self, input):
		activations = []

		x = input
		for m in self.nodeLevels:
			x = m(x)
			activations.append(x)

		return activations

squeezeNet = NetPerLayer()
squeezeNet = squeezeNet.cuda()
#squeezeNet =  torch.nn.DataParallel(squeezeNet, device_ids=[0, 1])
squeezeNet.eval()

def runDatasetIteration(dataset, isEval, itercount):

	epochTotalLoss = 0.0
	epochIterationCount = 0.0
	print('epoch start')
	repeatCount = 4

	meanToStart =5
	meanWeights = (np.linspace(0.0, 1.0, num=meanToStart+2)**2 )
	meanWeights = meanWeights[1:]
	lastNWeights = []

	for imgInput, imgTarget in dataset:
		for repeat in range(2):
	
			#print(torch.max(inputA))
			#torchvision.utils.save_image(imgInput, 'a.png', nrow=8, padding=2, normalize=True)
			#torchvision.utils.save_image(imgTarget, 'adist.png', nrow=8, padding=2, normalize=True)
			#quit()

			optimizer.zero_grad()

			img = Variable(imgInput.cuda())
			imgTarget = Variable(imgTarget.cuda())

			timeModelStart = time.time()	
			
			imgOutput = autoEncoder(img)

			activationsTarget = squeezeNet(imgTarget)
			activationsOutput = squeezeNet(imgOutput)

			featLoss = None
			#for actTarget, actOutput in zip(activationsTarget[1:3], activationsOutput[1:3]):
			for actTarget, actOutput in zip(activationsTarget, activationsOutput):
				#l = F.mse_loss(actTarget, actOutput)
				#l = torch.abs(actTarget - actOutput).sum()
				#l = F.l1_loss(actTarget, actOutput)

				l = -pearsonr(actTarget.view(-1), actOutput.view(-1))
				if featLoss is None:
					featLoss = l
				else:
					featLoss += l
			'''
			gradOutput = imgOutput[:,:, :-1,:-1] - imgOutput[:,:, 1:,1:]
			gradTarget = imgTarget[:,:, :-1,:-1] - imgTarget[:,:, 1:,1:]
			featLoss = F.l1_loss(gradOutput, gradTarget)
			'''
			pixelLoss = F.mse_loss(imgOutput, imgTarget )
			loss = featLoss +pixelLoss#+(pixelLoss*1)# + lossMSSIM# + reg_loss
			
			#loss = featLoss
			#loss.backward()

			#torch.nn.utils.clip_grad_norm_(autoEncoder.parameters(), 1)

			epochTotalLoss += loss.item()

			timeModelEnd = time.time()

			prefix = ''
			#isEval = True
			if isEval:
				# swap encoders
				prefix = 'EVAL: '
			else:	
				loss.backward()
				#print(loss)
				#with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
				#	scaled_loss.backward()
				optimizer.step()
			
			weights = autoEncoder.extract_parameters()
			lastNWeights.append(weights)
			if len(lastNWeights) > meanToStart:
				meanTeacher = np.array(lastNWeights)
				meanTeacher = np.average(meanTeacher, axis=0, weights=meanWeights).astype(np.float32)
				autoEncoder.inject_parameters(meanTeacher)
				del lastNWeights[0]
				
			#print("{} {}: total loss: {:.4f} feat: {:.4f} model-time: {}".format(prefix, 
			print("{} {} {}: total loss: {:.4f} (feat: {:.4f} pixel: {:.4f}) ... model-time: {:.2f}".format(prefix, 
				itercount, 
				repeat,
				loss.item(), 
				featLoss.item(),
				pixelLoss.item(),
				timeModelEnd-timeModelStart))

			itercount = itercount + 1
			epochIterationCount = epochIterationCount + 1

			if itercount == 1 or itercount % 100 == 0:

				#renderedImg = imgOutput.data.cpu()
				#renderedImg[imgInput != -1] = imgInput[imgInput != -1]
				#torchvision.utils.save_image(renderedImg.data.cpu(), './images/output-{}.png'.format(itercount), nrow=8, padding=2, normalize=True)

				outputTensor = torch.Tensor( imgOutput.size(0) * 3,
					imgOutput.size(1),
					imgOutput.size(2),
					imgOutput.size(3))

				b = 0
				for i in range(0, outputTensor.size(0), 3):
					outputTensor[i] = imgInput[b].data.cpu()
					outputTensor[i+1] = imgOutput[b].data.cpu()
					outputTensor[i+2] = imgTarget[b].data.cpu()
					b += 1
				
				torchvision.utils.save_image(outputTensor.data.cpu(), './images/output-{}.png'.format(itercount), nrow=3, padding=2, normalize=True)
		
	avgEpochLoss = epochTotalLoss / epochIterationCount
	return avgEpochLoss, itercount

while True:
	if evalFirst == False:
		print('starting epoch:')
		trainEpochLoss, itercount = runDatasetIteration(trainDataset, False, itercount)

		print('****** TRAINING EPOCH COMPLETE ... epoch loss: {} ******'.format(trainEpochLoss))

	#encoder.eval()
	#testEpochLoss, itercount = runDatasetIteration(trainDataset, True, itercount)
	#evalFirst = False

	#print('****** TESTING EPOCH COMPLETE ... epoch loss: {} ******'.format(testEpochLoss))
	#torch.save(model.state_dict(), './saves/mask-dist-eval-{}.pth'.format(testEpochLoss))

	print('saving models')
	#torch.save(autoEncoder.module.state_dict(), './saves/autoEncoder-{}.pth'.format(trainEpochLoss))
	torch.save(autoEncoder.state_dict(), './saves/autoEncoder-{}.pth'.format(trainEpochLoss))
	print('... save complete!')
	#torch.save(autoEncoder.state_dict(), './saves/autoEncoder-{}.pth'.format(trainEpochLoss))

