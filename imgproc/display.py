import numpy as np
import imutils
import cv2

class Display:
	def __init__(self, frameName, image, n, dim):
		# determine the OpenCV version,
		# for "motion" to be reported
		self.isv3 = imutils.is_cv3()

		# initialize the average image for motion detection
		self.myframe = frameName
		self.myimage = image
		self.myn = n
		self.mydim = dim
		(self.h, self.w) = image.shape[:2]
		
		if self.mydim == 1:
			self.out = np.zeros((self.h, self.w*self.myn), dtype="uint8")
		else:
			self.out = np.zeros((self.h, self.w*self.myn, self.mydim), dtype="uint8")
			


	def update(self, images):
		seq = 0
		for i in images:
			w1 = seq * self.w
			w2 = (seq+1) * self.w
			self.out[0:self.h, w1:w2] = i	
			seq = seq+1
		
		cv2.imshow( self.myframe, self.out)
		return
	