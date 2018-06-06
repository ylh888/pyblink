import imutils
import cv2
import queue

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
		self.list[0]= gray.copy().astype("float")
		self.list[1] = self.list[0].copy()
		if len(self.list)>2:
			self.list[2] = self.list[0].copy()
		if len(self.list)>3:
			self.list[3] = self.list[0].copy()
								 
#		for t in self.list:
#			t = gray.copy().astype("float")
	
	def findBest1(self, image, match_percentage):	
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
		matched = False
		if found[0] > match_percentage:
			cv2.accumulateWeighted(image, self.list[best], 0.1) # self.accumWeight)
			matched = True
		else:
			if lost[0] < 0.9:
				cv2.accumulateWeighted(image, self.list[worst], 0.8) # self.accumWeight)

		return (matched, found)

	
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

# find the top two in a competing set of n
# using the last q_size number of samples
# update(x) the new member
# returns (top, second)  if enough samples
# else returns(-1,-1) if x exceeds (0,n-1)
class Dominant:
	def __init__(self, xn, x_qsize):
		self.n = xn # number of elements
		self.q_size = x_qsize
		self.acc = [0] * self.n
		self.q = queue.Queue()
		self.cnt = 0
		
	def update(self, x):
		if x<0 or x>=self.n:
			return (-1,-1)

		# by default, goes from 0 to n-1
		self.cnt +=1
		self.acc[x] = self.acc[x] + 1
		self.q.put(x)
		if self.q.qsize() > self.q_size:
			y = self.q.get()
			self.acc[y] = self.acc[y] - 1
			return self.find_top_two()
		else: 
			return (-1,-1)

	def find_top_two(self):
		# https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
		
		inc_list = sorted(range(self.n), key=lambda k: self.acc[k])
		
		dec_list = inc_list[::-1] # reverse order
		
		return (dec_list[0], dec_list[1])
	
	def reset(self):
		self.acc = [0] * self.n
		self.q = queue.Queue()
		self.cnt = 0

	def tally(self):
		return self.acc
	
	def n_samples(self):
		return self.r
		

