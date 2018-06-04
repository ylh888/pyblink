import imutils
import cv2

class Match:
	def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		# determine the OpenCV version, followed by storing the
		# the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required
		# for "motion" to be reported
		self.isv3 = imutils.is_cv3()
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea

		# initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		return locs
	