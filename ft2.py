import cv2
import sys


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
ds_factor=0.6
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __dell__(self):
		self.video.release()

	def get_frame(self):
		ret, frame = self.video.read()

		frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30,30),
		)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		ret, jpeg = cv2.imencode('.jpg',frame)
		return jpeg.tobytes()

	


