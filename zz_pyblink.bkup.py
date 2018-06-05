# USAGE
# python webcam_fall.py --conf conf.json | confShow.json
# derived from << webcam_surv.py and webcam_fall.py
# "system" = "OSX" or "PI"

# import the necessary packages
from pyimagesearch.tempimage import TempImage
from imgproc import Display
from imgproc import Match
from imgproc import PreProc

import numpy as np
import dropbox

import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2

from imutils.video import VideoStream

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
ap.add_argument("-v", "--video", required=False,
	help="path to input video file")
args = vars(ap.parse_args())

# open a pointer to the video stream and start the FPS timer
fromVideo = False
if args["video"]:
	fromVideo = True


# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %a") #"%A %d %B %Y %I:%M:%S%p")
print("===")
print(ts)

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print( "[SUCCESS] dropbox account linked" )


# initialize the camera and grab a reference to the raw camera capture
if fromVideo:
	print("Open from:", args["video"])
	stream = cv2.VideoCapture(args["video"])
else:
	os = "OSX"
	if conf["system"]:
		os = conf["system"]
		if os == "OSX":
			camera = cv2.VideoCapture(0)
		if os == "PI":
			# usePiCamera=True OR src=0 if using USB camera
			picam = VideoStream(usePiCamera=True).start()
			from picamera.array import PiRGBArray
			from picamera import PiCamera

if fromVideo:
	(grabbed, frame) = stream.read()
else:
	if os == "OSX":
		(grabbed, frame) = camera.read()
	else:
		frame = picam.read()
		
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print( "[INFO] warming up..." )
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()

#### experiment with preset parameters	
scaled_height = 1000
if conf["scaled_height"]:
	scaled_height = conf["scaled_height"]
if conf["min_area"]:
	min_area = conf["min_area"]
	
# >>>>over-write with scaled_height
# heuristic for min area for any blob
min_area =int(scaled_height/12) * int(scaled_height/8)

# all blobs aggregated should have the following minimums
min_w = int(scaled_height/8)
min_h = int(scaled_height/3)


motionCounter = 0
fall_detected = 0
ytop = 0
yfloor = 0
fall_threshold = 0
thisx = 0
thisy = 0
cnt = -1
graph = None
minX=minY=maxX=maxY=0
fall_noted = False
text_color = (255,0,0)

ntemplates = 3



# resize the frame, convert it to grayscale, and blur it
frame = imutils.resize(frame, height=scaled_height)

preproc = PreProc(scaled_height)
gray = preproc.proc1(frame)
templates = [gray] * ntemplates
displayGray = Display("TEMPLATES", templates, gray, ntemplates, 1)
displayGray.update()

match = Match() 

# capture frames from the camera
#for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:
	#time.sleep(0.01)  # ylh delay
	cnt += 1

	# grab the raw NumPy array representing the image and initialize
	if fromVideo:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
	else:
		if os == "OSX":
			(grabbed, frame) = camera.read()
			if not grabbed:
				break
		else:
			frame = picam.read()

	timestamp = datetime.datetime.now()
	text = "Still"
	
	# resize the frame, convert it to grayscale, and blur it
	cv2.imshow("Orig", frame)

	gray = preproc.proc1(frame)
	cv2.imshow("Gray", gray)
	
	templates[1] = gray
	displayGray.update()
	
	# if the average frame is None, initialize it
	if avg is None:
		print( "[INFO] starting background model...")
		avg = gray.copy().astype("float")
		graph = imutils.resize(frame, height=50)
		graph = imutils.resize(graph, height=scaled_height)
		#rawCapture.truncate(0)
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average

	# cv2.accumulateWeighted(gray, avg, 0.5)  # ORIGINAL PLACE HERE ylh
	cv2.accumulateWeighted(gray, avg, 0.5) # do greater update when not moving
	
	# convertScaleAbs(src,dest, alpha, beta) Scales, calculates absolute values, and converts the result to 8-bit.
	# Python: cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst

	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
	
	# heuristic for top
	thisy = scaled_height # modified height
	
	locs = []
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < min_area: # conf["min_area"]:
			continue
		
		locs.append(c)
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
		#cv2.rectangle(gray, (x, y), (x + w, y + h), (255,255,255), 1)
		cv2.rectangle(thresh, (x, y), (x + w, y + h), (255,255,255), 1)
		cv2.rectangle(thresh, (x, y), (x + min_w, y + min_h), (255,255,255), 1)

	if len(locs) > 0:
		# initialize the minimum and maximum (x, y)-coordinates,
		# respectively
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# loop over the locations of motion and accumulate the
		# minimum and maximum locations of the bounding boxes
		for l in locs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))
		if (maxX-minX) < min_w or (maxY-minY) < min_h:
			continue

		# draw the bounding box
		cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 255, 0), 1)
		cv2.rectangle(thresh, (minX, minY), (maxX, maxY),(255,255,255), 1)

		text = "Movement"
		# print("x,y %d %d w=%d h=%d a=%dk" % (x,y,w,h, w*h/1000))
		# fall detection - top and floor
		if ytop == 0:
			ytop = minY
		alpha = 0.05
		ytop = ytop*(1-alpha)+ minY*alpha
		if yfloor == 0:
			yfloor = maxY
		yfloor = yfloor*(1-alpha)+ maxY*alpha

		#===== set fall_threshold
		fall_threshold = int(round(ytop + (yfloor-ytop)*.3))

		# set the current origin
		thisy = minY
		thisx = minX
		# graph fall_threshold, h/w
		f=fall_threshold
		cv2.circle(graph, (cnt, f), 2, (0,255,0), 1)
		f= scaled_height - int(round( (maxY-minY)*scaled_height/(maxX-minX)/8) )
		if (maxX-minX) < (maxY-minY):
			color = (0,255,255)
		else:
			color = (0,0,255)  # red when w>h
		cv2.circle(graph, (cnt, f), 2, color, 1)

	# check to see if the room is occupied
	if text == "Movement":
		# reset by touching the edge
		if( thisx < 40 ):
			fall_detected = 0
			fall_noted = False
			ytop = yfloor = 0
			avg = gray.copy().astype("float") # ylh
			cv2.circle(graph, (cnt, 5), 3, (200,80,0),2)
			
		
		# threshold trigger
		# h/w ratio trigger
		if ( thisy > fall_threshold ): # or (((maxY-minY)*100/(maxX-minX)) < 100):
			fall_detected  += 1
			
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1

			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				# check to see if dropbox sohuld be used
				if conf["use_dropbox"]:
					# write the image to temporary file
					t = TempImage()
					cv2.imwrite(t.path, frame)

					# upload the image to Dropbox and cleanup the tempory image
					if conf["show_debug"]:
						print( "[UPLOAD] {}".format(ts) )
					path = "/{base_path}/{timestamp}.jpg".format(
						base_path=conf["dropbox_base_path"], timestamp=ts)
					client.files_upload(open(t.path,"rb").read(),path)
					t.cleanup()

				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0

	# otherwise, the room is not occupied
	else:
		motionCounter = 0
		cv2.accumulateWeighted(gray, avg, 0.0)  # we only update when no movement

	if fall_detected > 0:
		text_color = (0,0,255);
	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a") #"%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Status: {}".format(text), (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, text_color, 1)
	
	# draw the top, floor and threshold lines
	cv2.rectangle(thresh, (0, (int(round(ytop)))), (int(round(scaled_height*1.8)), (int(round(ytop)))), (80, 155, 80), 1)
	cv2.rectangle(thresh, (0, (int(round(yfloor)))), (int(round(scaled_height*1.8)), (int(round(yfloor)))), (80, 155, 80), 1)	
	cv2.rectangle(thresh, (0, (int(round(fall_threshold)))), (int(round(scaled_height*1.8)), (int(round(fall_threshold)))), (80, 80, 155), 2)
	
	if fall_detected > 0:
		text_color = (0,0,255);
		
		text = "Fall Detected"
		if fall_detected == 1:
			if fall_noted == False: # do this once per fall detected
				fall_noted = True
				# vertical event bar
				cv2.rectangle(graph, (cnt, 0), (cnt, scaled_height), (0,0,155), 1)
	#			print(" ")
				print(" %d * w=%d h=%d h/w=%d a=%dk thresh=%d" % ( fall_detected, w,h, int(h*100/w), w*h/1000, fall_threshold ) )
				if conf["save_local"]:
					# write the image to temporary file
					ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a 0.jpg")
					cv2.imwrite(ts, frame)
					ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a t.jpg")
					cv2.imwrite(ts, thresh)
					
		h=maxY-minY
		w=maxX-minX
		# set text and the cancel bar
		cv2.putText(frame, "FALL DETECTED", (int(scaled_height/6), int(scaled_height/2)),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
		cv2.rectangle(frame, (0, 0), (10, scaled_height), (0,0,255), 20)
	elif text == "Still":
		text_color = (255, 0, 0);		
	else:
		text_color = (0, 255, 0);		

	# reset graph display
	if cnt >= int(round(scaled_height*1.7)):
		cnt = 0
		(grabbed, graph) = camera.read()
		graph = imutils.resize(graph, height=50)
		graph = imutils.resize(graph, height=scaled_height)
		fall_detected = 0
		ytop = yfloor = 0
		avg = gray.copy().astype("float") # ylh
		cv2.circle(graph, (cnt, 5), 3, (200,80,0),2)

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
#		cv2.imshow("delta", frameDelta)

#		cv2.imshow("gray", gray)
		cv2.imshow("thresh", thresh)
		cv2.imshow("Graph", graph)
		cv2.imshow("Avg", cv2.convertScaleAbs(avg) )
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

	# clear the stream in preparation for the next frame
	#rawCapture.truncate(0)
