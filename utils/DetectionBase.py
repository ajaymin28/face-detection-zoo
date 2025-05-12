class DetectionBase:
    def __init__(self,upSample=0, frame_queue_size=1):
        self.face_locations = []
        self.frame_queue = []
        self.frame_queue_size = frame_queue_size
        self.detected_frame = None
        self.upSample = upSample

    def setInput(self, inputFrame):
        self.face_locations = []
        if len(self.frame_queue)>=self.frame_queue_size:
            self.frame_queue.pop(0)
        self.frame_queue.append(inputFrame)

    def getOutput(self):

        pass