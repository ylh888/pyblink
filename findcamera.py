import cv2
import time

for i in range(10):
#    capture = cv2.CaptureFromCAM(i)
	camera = cv2.VideoCapture(i)
	if camera:
		(grabbed, frame) = camera.read()
		if grabbed:
			cv2.imshow("video source " + str(i), frame)
			
			
key = cv2.waitKey(1) & 0xFF

time.sleep(30)
## if the `q` key is pressed, break from the lop
#if key == ord("q"):
#	break
