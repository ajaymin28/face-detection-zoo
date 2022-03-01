from utils.HaarCascadeFaceDetection import  HaarCascadeFaceDetection
from utils.MediaPipeFaceDetection import MediaPipeFaceDetection
from utils.DLibFaceDetection import DLibFaceDetection
import cv2

DlibFace = DLibFaceDetection()
HaarFace = HaarCascadeFaceDetection()
MediaPipeFace = MediaPipeFaceDetection()


# cap_webCam = cv2.VideoCapture("rtsp://192.168.0.111:554/live/ch00_0")
cap = cv2.VideoCapture(0)


InputFrameWindowName = "InputFrame Window"
DlibFrameWindowName  = "Dlib Window"
WebCamViewWindowName = "WebCam Window" 
HaarCasecaseWindowName = "Haar Face Detection"
MediaPipeWindowName  = "Media Pipe Face Detection"

cv2.namedWindow(InputFrameWindowName)
cv2.namedWindow(DlibFrameWindowName)
# cv2.namedWindow(WebCamViewWindowName)
cv2.namedWindow(HaarCasecaseWindowName)
cv2.namedWindow(MediaPipeWindowName)

while cap.isOpened():
    success, image = cap.read()
    if success:

        DlibFace.setInput(image.copy())
        MediaPipeFace.setInput(image.copy())
        HaarFace.setInput(image.copy())

        DlibOutputFrame = DlibFace.getOutput()
        if DlibOutputFrame is not None:
            cv2.imshow(DlibFrameWindowName, DlibOutputFrame)

        HaarCascadeOutputFrame = HaarFace.getOutput()
        if HaarCascadeOutputFrame is not None:
            cv2.imshow(HaarCasecaseWindowName, HaarCascadeOutputFrame)

        MediaPipeOutputFrame = MediaPipeFace.getOutput()
        if MediaPipeOutputFrame is not None:
            cv2.imshow(MediaPipeWindowName, MediaPipeOutputFrame)
        
        cv2.imshow(InputFrameWindowName, image)

    # webcam_success, webcam_image = cap_webCam.read()
    # if webcam_success:
    #     cv2.imshow(WebCamViewWindowName, webcam_image)




    if cv2.waitKey(5) & 0xFF == 27:
        MediaPipeFace.exit_thread()
        HaarFace.exit_thread()
        DlibFace.exit_thread()
        break



    






        


