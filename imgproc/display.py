import numpy as np
import imutils
import cv2

class Display: #### can only handle list of images
	# use ximage as template for h,w
	# xlist is the image to update
	def __init__(self, xframeName, xlist, ximage, xn, xdim):
		# determine the OpenCV version,
		# for "motion" to be reported
		self.isv3 = imutils.is_cv3()

		# initialize the average image for motion detection
		self.frameName = xframeName
		self.list = xlist
		self.n = xn
		self.dim = xdim
		(self.h, self.w) = ximage.shape[:2]
		
		if self.dim == 1:
			self.out = np.zeros((self.h, self.w*self.n), dtype="uint8")
		else:
			self.out = np.zeros((self.h, self.w*self.n, self.dim), dtype="uint8")
			


	def update(self, drawwhich=0, drawmarker=False):
		seq = 0
		for i in self.list:
			w1 = seq * self.w
			w2 = (seq+1) * self.w
			self.out[0:self.h, w1:w2] = i
			# draw a border
			if drawmarker and drawwhich==seq:
				cv2.rectangle(self.out, (w1+4, 4), (w2-6,self.h-4), (0,255,0), 2)
			seq = seq+1
		
		cv2.imshow( self.frameName, self.out)
		return
	