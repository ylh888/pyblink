# USAGE
# python webcam_fall.py --conf conf.json | confShow.json
# derived from << webcam_surv.py and webcam_fall.py
# "system" = "OSX" or "PI"

# import the necessary packages
from pyimagesearch.tempimage import TempImage
from imgproc import Display
from imgproc import Match
from imgproc import PreProc
from imgproc import Dominant

import numpy as np
import dropbox

import argparse
import warnings
import datetime
import time
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
os = "OSX"
if fromVideo:
	print("Open from:", args["video"])
	stream = cv2.VideoCapture(args["video"])
else:
	if conf["system"]:
		os = conf["system"]
		if os == "OSX":
			if conf["camera_source"]:
				stream = cv2.VideoCapture(conf["camera_source"] ) 
			else:
				stream = cv2.VideoCapture(0)
		if os == "PI":
			# ************************ WATCH THIS
			# usePiCamera=True OR src=0 if using USB camera
			stream = VideoStream(usePiCamera=True).start()
			# stream = VideoStream(0).start()
			from picamera.array import PiRGBArray
			from picamera import PiCamera

if fromVideo:
	(grabbed, frame) = stream.read()
else:
	if os == "OSX":
		(grabbed, frame) = stream.read()
	else:
		frame = stream.read()

		
match_percentage = 0.9
if conf["match_percentage"]:
	match_percentage = conf["match_percentage"]
nmatch_criterion = 10
if conf["nmatch_criterion"]:
	nmatch_criterion = conf["nmatch_criterion"]
cycle_time = 0.05 # every 1/20 of a second
if conf["cycle_time"]:
	cycle_time = conf["cycle_time"]
	
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
blink_detected = 0
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

ntemplates = 2
templates = []
last_matched = -1 
nseqmatches = 0

# resize the frame, convert it to grayscale, and blur it
frame = imutils.resize(frame, height=scaled_height)
preproc = PreProc(scaled_height)
gray = preproc.proc1(frame)

#templates = [avg] * ntemplates
for x in range(0, ntemplates):
	templates.append(gray.copy().astype("float"))

print(len(templates))

displayTemplates= Display("TEMPLATES", templates, gray.copy().astype("float"), ntemplates, 1)
displayTemplates.update()

match = Match(templates, accumWeight=0.05)  
match.findBest1(gray, match_percentage)

dom = Dominant(ntemplates, 100)

# capture frames from the camera
#for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
last_time = time.time()
while True:
	if( (time.time() - last_time) < cycle_time): # every n seconds
		continue
	last_time = time.time()
	
	timestamp = datetime.datetime.now()
	cnt += 1  # frame count
	text = "Still"

# grab the raw NumPy array representing the image and initialize
#	if fromVideo:
#		(grabbed, frame) = stream.read()
#		if not grabbed:
#			break
#	else:
	if os == "OSX" or fromVideo:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
	else:
		frame = stream.read()

	
	gray = preproc.proc1(frame)
	# found = (maxVal, maxLoc, seq)
	(matched, found) = match.findBest1(gray, match_percentage)
	
#	print(found)
#	if found[0] > match_percentage:
#		matched = True

	if conf["show_video"]:
#		cv2.imshow("gray", gray)
		displayTemplates.update(found[2], matched)
		if blink_detected>0:
			cv2.rectangle(frame, (0,0), (40, frame.shape[0]), (80, 155, 80), 40)
		cv2.imshow("Original", frame)
	
	# heuristics: how many consecutive matches of a template
	# if this exceeds nmatch_criterion, it is a 'hit' 
	# then we check if the template is
	# the dominant one in the last x number of hits
	# the NON dominant one would trigger a blink_detected
	if found[2]==last_matched:  # which template is matched?
		nseqmatches += 1
		if nseqmatches > nmatch_criterion:
			result = dom.update(found[2])
	
			if result[0]>=0 and last_matched != result[0]:  # non-dominant
				blink_detected += 1
			if blink_detected == 1:
				# the first time
				print("BLINKED", cnt, int((time.time() -time_first_matched)*1000)/1000, dom.tally() )
				print('\a')
				
				
			if blink_detected >1:
				# ignore the rest of the time
				blink_detected = blink_detected # no op
	else:
		last_matched = found[2]
		nseqmatches = 1
		blink_detected = 0
		time_first_matched = time.time()
 

	# after nseqmatches, we need to ascertain if this is the
	# "normal" state or "blink" state by examine the dominant cnt
	
	if text == "Blinked":
		# reset by touching the edge
		if( thisx < 40 ):
			blink_detected = 0
			fall_noted = False
			ytop = yfloor = 0
			avg = gray.copy().astype("float") # ylh
			cv2.circle(graph, (cnt, 5), 3, (200,80,0),2)
			
		
		# threshold trigger
		# h/w ratio trigger
		if ( thisy > fall_threshold ): # or (((maxY-minY)*100/(maxX-minX)) < 100):
			blink_detected  += 1
			
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
#		cv2.accumulateWeighted(gray, avg, 0.0)  # we only update when no movement

	if blink_detected > 0:
		text_color = (0,0,255);
		
	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a") #"%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Status: {}".format(text), (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, text_color, 1)
	
	# draw the top, floor and threshold lines
	#	cv2.rectangle(thresh, (0, (int(round(ytop)))), (int(round(scaled_height*1.8)), (int(round(ytop)))), (80, 155, 80), 1)
	#	cv2.rectangle(thresh, (0, (int(round(yfloor)))), (int(round(scaled_height*1.8)), (int(round(yfloor)))), (80, 155, 80), 1)	
	#	cv2.rectangle(thresh, (0, (int(round(fall_threshold)))), (int(round(scaled_height*1.8)), (int(round(fall_threshold)))), (80, 80, 155), 2)
	
#	if blink_detected > 0:
#		text_color = (0,0,255);
#		
#		text = "Blink Detected"
#		if blink_detected == 1:
#			if fall_noted == False: # do this once per fall detected
#				fall_noted = True
#				# vertical event bar
#				cv2.rectangle(graph, (cnt, 0), (cnt, scaled_height), (0,0,155), 1)
#	#			print(" ")
#				print(" %d * w=%d h=%d h/w=%d a=%dk thresh=%d" % ( blink_detected, w,h, int(h*100/w), w*h/1000, fall_threshold ) )
#				if conf["save_local"]:
#					# write the image to temporary file
#					ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a 0.jpg")
#					cv2.imwrite(ts, frame)
#					ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a t.jpg")
#					cv2.imwrite(ts, thresh)
#					
#		h=maxY-minY
#		w=maxX-minX
#		# set text and the cancel bar
#		cv2.putText(frame, "FALL DETECTED", (int(scaled_height/6), int(scaled_height/2)),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
#		cv2.rectangle(frame, (0, 0), (10, scaled_height), (0,0,255), 20)
#	elif text == "Still":
#		text_color = (255, 0, 0);		
#	else:
#		text_color = (0, 255, 0);		

	# reset graph display
	#	if cnt >= int(round(scaled_height*1.7)):
	#		cnt = 0
	#		(grabbed, graph) = camera.read()
	#		graph = imutils.resize(graph, height=50)
	#		graph = imutils.resize(graph, height=scaled_height)
	#		blink_detected = 0
	#		ytop = yfloor = 0
	#		avg = gray.copy().astype("float") # ylh
	#		cv2.circle(graph, (cnt, 5), 3, (200,80,0),2)

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
#		displayTemplates.update()
#		# display the security feed
#		cv2.imshow("Security Feed", frame)
##		cv2.imshow("delta", frameDelta)
#
#		cv2.imshow("thresh", thresh)
#		cv2.imshow("Graph", graph)
#		cv2.imshow("Avg", cv2.convertScaleAbs(avg) )
		key = cv2.waitKey(1) & 0xFF

		# re-start
		if key == ord("r"):
			if os == "OSX" or fromVideo:
				(grabbed, frame) = stream.read()
				if not grabbed:
					break
			else:
				frame = stream.read()

			gray = preproc.proc1(frame)
			match.reset(gray)
			dom.reset()
			blink_detected = 0
			nseqmatches = 0
			
			
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break


	# clear the stream in preparation for the next frame
	#rawCapture.truncate(0)
