#from pycocotools.coco import COCO
import numpy as np
import skimage.io as io

import torch
from PIL import Image, ImageDraw
import torch as t
from torchvision.transforms import ToPILImage
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import scipy.misc
import cv2
import random
import os
import scipy
from scipy import ndimage
import math
import random
import torch.nn.functional as F


#from imgaug import augmenters as iaa
from scipy import ndimage as ndi
from skimage import feature

from scipy.spatial import distance
class ABDataloader(Dataset):

	def __init__(self, opt):

		self.opt = opt     
		
		self.imgPathsA = self.buildDataPaths(opt.dataroota)
		
		#self.detector = dlib.get_frontal_face_detector()
		#self.predictor = dlib.shape_predictor('/home/lolz0r/domain/vae/shape_predictor_68_face_landmarks.dat')



	def buildDataPaths(self, dataroots):

		results = []

		for dataroot in dataroots:
			print('scanning ' + dataroot)
			for root, dirs, files in os.walk(dataroot, topdown=True):
				for name in files:
					path = os.path.join(root, name)

					if '.jpg' in path and '.cached' not in path:
						results.append(path)
					
			print(len(results))
		return results

	def loadRandomImage(self, dataset):
		idx = random.randint(0, len(dataset)-1)
		path = dataset[idx]
	
		#print(path)
		#img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YCrCb).copy()
		img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).copy()
		#img = cv2.imread(path)
	
		if img.shape[0] < img.shape[1]:
			r = float(img.shape[1]) / float(img.shape[0])
			s = int(float(self.opt.loadSize) * r)
			img = cv2.resize(img, (s, self.opt.loadSize))
		else:
			r = float(img.shape[0]) / float(img.shape[1])
			s = int(float(self.opt.loadSize) * r)
			img = cv2.resize(img, (self.opt.loadSize, s))
			

		y = random.randint(0, img.shape[0]  - (self.opt.imgSize + 1))
		x = random.randint(0, img.shape[1]  - (self.opt.imgSize + 1)) 
		img = img[y:y+self.opt.imgSize, x:x+self.opt.imgSize]

		#img = cv2.resize(img, dsize= None, fx=0.5, fy=0.5)
		croppedImg = np.copy(img)

		#croppedImg = cv2.Laplacian(croppedImg,cv2.CV_64F)
		#croppedImg = cv2.Canny(croppedImg,1,255)
		#cannyImg =  feature.canny(croppedImg[:,:,0])

		shrinkImg = cv2.resize(cv2.resize(img, (16,16)), (self.opt.imgSize, self.opt.imgSize))

		for i in range(3):
			croppedImg[:,:,i] = cv2.Canny(croppedImg[:,:,i], 1, 255)
		idxOfZeroCanny = croppedImg == 0	
		croppedImg[idxOfZeroCanny] = shrinkImg[idxOfZeroCanny]

		'''
		imgBW =  cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
		imgBWCanny =  cv2.Canny(imgBW, 1, 255)
		idxOfZeroCanny = imgBWCanny == 0
		for i in range(3):
			croppedImg[:,:,i] = imgBWCanny
			
		croppedImg[idxOfZeroCanny, :] = shrinkImg[idxOfZeroCanny, :]
		'''

		#subslice = croppedImg[cropX:cropX+patchRemovalW, cropY:cropY+patchRemovalH,:] 
		#subslice = np.random.uniform(0, 255, size=subslice.shape)
		#croppedImg[cropX:cropX+patchRemovalW, cropY:cropY+patchRemovalH,:] = subslice

		croppedImg =  (((croppedImg.astype(np.float32)) / 255.0) * 2) - 1
		croppedImg = torch.from_numpy(croppedImg.astype(np.float32) )
		croppedImg = croppedImg.permute(2, 0, 1)

		# normalize
		img =  (((img.astype(np.float32)) / 255.0) * 2) - 1
		img = torch.from_numpy(img.astype(np.float32) )
		img = img.permute(2, 0, 1)

		return img, croppedImg

	def __getitem__(self, index):

		imgA, croppedImgA = self.loadRandomImage(self.imgPathsA)
	
		return croppedImgA, imgA
		#return cellImg, maskOuput.astype(np.int64), distance

	def __len__(self):
		return self.opt.itersPerEpoch 
