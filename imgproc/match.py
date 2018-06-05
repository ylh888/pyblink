import imutils
import cv2

class Match:
	def __init__(self, templates, accumWeight=0.5, deltaThresh=5, minArea=5000, cropSize = 10):
		# determine the OpenCV version, followed by storing the
		# the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required
		# for "motion" to be reported
		self.isv3 = imutils.is_cv3()
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea
		self.cropSize = cropSize
		
		self.list = templates
		(self.h, self.w) = templates[0].shape[:2]

	def reset (self, gray):
		print("RESET===========================")
		self.list[2] = self.list[0].copy()
		self.list[1] = self.list[0].copy()
								 
#		for t in self.list:
#			t = gray.copy().astype("float")
	
	def findBest1(self, image):	
		found = None  # the best matched
		best = 0
		maxVal = 0
		lost = None  # the worst matched
		worst = 0
		minVal = 0
	
		
		cropped = image[self.cropSize:(self.h-self.cropSize), self.cropSize:(self.w-self.cropSize)]
		
		seq=0
		for t in self.list:
			result = cv2.matchTemplate(image, cv2.convertScaleAbs(t), cv2.TM_CCOEFF_NORMED)
			
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 	
			# if we have found a new maximum correlation value, then update
			if found is None or maxVal > found[0]:
				best = seq
				found = (maxVal, maxLoc, seq)
			#worst
			if lost is None or minVal < lost[0]:
				worst = seq
				lost = (maxVal, maxLoc, seq)
			seq=seq+1
		
		#update the best matched template
		if found[0] > 0.90:
			cv2.accumulateWeighted(image, self.list[best], 0.1) # self.accumWeight)
		else:
			if lost[0] < 0.6:
				cv2.accumulateWeighted(image, self.list[worst], 0.8) # self.accumWeight)

		return found
	
	
class PreProc:
	def __init__(self, xscaled_height):
		self.scaled_height = xscaled_height
	
	def proc1(self, img):
		# resize, make gray, blur
		gray = imutils.resize(img, height=self.scaled_height)
		
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	#	gray = cv2.GaussianBlur(gray, (21, 21), 0)
		gray = cv2.GaussianBlur(gray, (3,3), 0)
		return gray