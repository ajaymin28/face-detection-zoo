import mediapipe as mp
import threading


class MediaPipeFaceDetection():

	def __init__(self,frame_queue_size=1):
		self.mp_face_detection = mp.solutions.face_detection
		self.mp_drawing = mp.solutions.drawing_utils
		self.frame_queue = []
		self.frame_queue_size = frame_queue_size
		self.frame_queue_lock = threading.Lock()
		self.detected_mediapipe_frame = None

		self.t1 = threading.Thread(target=self.detect)
		self.t1.setName("MediaPipe Detector Thread")
		self.runThread = True
		self.t1.start()
		
	def __del__(self):
		self.t1.join()

	def setInput(self, inputFrame):
		self.frame_queue_lock.acquire()
		# Keep latest frames of size [self.frame_queue_size]
		if len(self.frame_queue)>=self.frame_queue_size:
			self.frame_queue.pop(0)
		self.frame_queue.append(inputFrame)
		self.frame_queue_lock.release()

	def getOutput(self):
		self.frame_queue_lock.acquire()
		finalFrame = self.detected_mediapipe_frame
		self.frame_queue_lock.release()
		return finalFrame

	def exit_thread(self):
		self.runThread = False
		self.__del__()


	def detect(self):

		while self.runThread:
			inputFrame = None
			self.frame_queue_lock.acquire()
			if len(self.frame_queue)>0:
				inputFrame = self.frame_queue.pop(0)
			self.frame_queue_lock.release()


			if inputFrame is not None:
				with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
					results = face_detection.process(inputFrame)
					if results.detections:
						for detection in results.detections:
							self.mp_drawing.draw_detection(inputFrame, detection)

			self.frame_queue_lock.acquire()
			self.detected_mediapipe_frame = inputFrame
			self.frame_queue_lock.release()
