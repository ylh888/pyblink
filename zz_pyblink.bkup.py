# USAGE
# python pyblink.py --conf conf.json
# derived from << webcam_surv.py
# "system" = "OSX" or "PI"

# import the necessary packages
from pyimagesearch.tempimage import TempImage
import numpy as np
import dropbox
#from picamera.array import PiRGBArray
#from picamera import PiCamera
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
args = vars(ap.parse_args())

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
		
os = "OSX"
if conf["system"]:
	os = conf["system"]
	if os == "PI":
		from picamera.array import PiRGBArray
		from picamera import PiCamera
# initialize the camera and grab a reference to the raw camera capture
if os == "OSX":
	camera = cv2.VideoCapture(0)
else:
	# usePiCamera=True OR src=0 if using USB camera
	picam = VideoStream(usePiCamera=True).start()

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print( "[INFO] warming up..." )
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
#for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:
	time.sleep(0.05)  # ylh delay

	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	#--frame = f.array
	if os == "OSX":
		(grabbed, frame) = camera.read()
		if not grabbed:
			break
	else:
		frame = picam.read()
	timestamp = datetime.datetime.now()
	text = "Unoccupied"

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the average frame is None, initialize it
	if avg is None:
		print( "[INFO] starting background model...")
		avg = gray.copy().astype("float")
		#rawCapture.truncate(0)
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
		text = "Occupied"

	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%Y-%m-%d %H:%M:%S %a") #"%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# check to see if the room is occupied
	text == "Occupied" # FORCED SENDING IMAGES to DROPBOX
	if text == "Occupied":
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
					#path = "{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
					path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
					#client.put_file(path, open(t.path, "rb"))
					client.files_upload(open(t.path,"rb").read(),path)
					t.cleanup()

				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0

	# otherwise, the room is not occupied
	else:
		motionCounter = 0

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

	# clear the stream in preparation for the next frame
	#rawCapture.truncate(0)
