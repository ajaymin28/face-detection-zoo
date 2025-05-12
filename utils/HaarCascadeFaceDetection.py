import cv2
import os
import threading

class HaarCascadeFaceDetection():

	def __init__(self, frame_queue_size=1):
		self.haar_frame_queue = []
		self.haar_frame_queue_size = frame_queue_size
		self.haar_frame_queue_lock = threading.Lock()
		self.detected_haarcascade_frame = None
		self.haarcasecadefile = os.path.join("files") + "/haarcascade_frontalface_alt.xml"
		self.face_cascade = cv2.CascadeClassifier()
		if not self.face_cascade.load(cv2.samples.findFile(self.haarcasecadefile)):
			print('--(!)Error loading face cascade')

		self.t1 = threading.Thread(target=self.detect)
		self.t1.setName("HaarCascade Detector Thread")
		self.runThread = True
		self.t1.start()


	def __del__(self):
		self.t1.join()

	def setInput(self, inputFrame):
		self.haar_frame_queue_lock.acquire()
		# Keep latest frames of size [self.frame_queue_size]
		if len(self.haar_frame_queue)>=self.haar_frame_queue_size:
			self.haar_frame_queue.pop(0)
		self.haar_frame_queue.append(inputFrame)
		self.haar_frame_queue_lock.release()

	def getOutput(self):
		self.haar_frame_queue_lock.acquire()
		finalFrame = self.detected_haarcascade_frame
		self.haar_frame_queue_lock.release()
		return finalFrame


	def exit_thread(self):
		self.runThread = False
		self.__del__()

	def detect(self):

		while self.runThread:
			inputFrame = None
			self.haar_frame_queue_lock.acquire()
			if len(self.haar_frame_queue)>0:
				inputFrame = self.haar_frame_queue.pop(0)
			self.haar_frame_queue_lock.release()


			if inputFrame is not None:

				frame_gray = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
				frame_gray = cv2.equalizeHist(frame_gray)

				faces = self.face_cascade.detectMultiScale(frame_gray)
				for (x,y,w,h) in faces:
					center = (x + w//2, y + h//2)
					inputFrame = cv2.ellipse(inputFrame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

			self.haar_frame_queue_lock.acquire()
			self.detected_haarcascade_frame = inputFrame
			self.haar_frame_queue_lock.release()



