import dlib
import threading
import cv2


class DLibFaceDetection():

	def __init__(self, upSample=0, frame_queue_size = 1):
		self.dlib_face_detector =  dlib.get_frontal_face_detector()
		self.frame_queue = []
		self.frame_queue_size = frame_queue_size
		self.dlib_queue_lock = threading.Lock()
		self.detected_frame = None
		self.upSample = upSample
		self.t1 = threading.Thread(target=self.detect)
		self.t1.setName("Dlib Detector Thread")
		self.runThread = True
		self.t1.start()

	
	def __del__(self):
		self.t1.join()

	def setInput(self, inputFrame):
		self.dlib_queue_lock.acquire()
		# Keep latest frames of size [self.frame_queue_size]
		if len(self.frame_queue)>=self.frame_queue_size:
			self.frame_queue.pop(0)
		self.frame_queue.append(inputFrame)
		self.dlib_queue_lock.release()

	def getOutput(self):
		self.dlib_queue_lock.acquire()
		finalFrame = self.detected_frame
		self.dlib_queue_lock.release()
		return finalFrame


	def exit_thread(self):
		self.runThread = False
		self.__del__()

	def detect(self):

		while self.runThread:
			inputFrame = None
			self.dlib_queue_lock.acquire()
			if len(self.frame_queue)>0:
				inputFrame = self.frame_queue.pop(0)
			self.dlib_queue_lock.release()


			if inputFrame is not None:
				dets = self.dlib_face_detector(inputFrame, self.upSample)
				for i, rect in enumerate(dets):

					startX = rect.left()
					startY = rect.top()
					endX = rect.right()
					endY = rect.bottom()

					startX = max(0, startX)
					startY = max(0, startY)
					endX = min(endX, inputFrame.shape[1])
					endY = min(endY, inputFrame.shape[0])

					w = endX - startX
					h = endY - startY

					cv2.rectangle(inputFrame, (startX, startY), (startX + w, startY + h), (0, 255, 0), 2)
					
			self.dlib_queue_lock.acquire()
			self.detected_frame = inputFrame
			self.dlib_queue_lock.release()