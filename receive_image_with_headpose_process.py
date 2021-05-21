from openvino.inference_engine import IECore
from head_pose import HeadPose
from face_detection import FaceDetection
import cv2
import numpy as np


#path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

HEADPOSE_XML_PATH = "./models/head-pose-estimation-adas-0001.xml"
HEADPOSE_BIN_PATH = "./models/head-pose-estimation-adas-0001.bin"


#Cam rtsp link
CAM_ADDR = "rtsp://ftech:ad1235min@192.168.130.27/live0"

ie = IECore()

#Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

#Create HeadPose model
headpose = HeadPose(ie, HEADPOSE_XML_PATH, HEADPOSE_BIN_PATH)

#init video from rtsp link
video = cv2.VideoCapture(CAM_ADDR)

#init video from build-in camera
# video = cv2.VideoCapture(0)

while(video.isOpened()):
    ret, frame = video.read()
    show_frame = frame.copy()
    h, w, c = frame.shape

    #get face on frame
    outputs = facedetection.detect(frame)

    if len(outputs) != 0:
        outputs = np.array(outputs)
        for output in outputs:
            try:
                #get face location
                x_min, y_min, x_max, y_max = (output * [w, h, w, h]).astype(int)

                #crop face
                img_cropped = frame[y_min:y_max, x_min:x_max]

                #get face tilt
                yaw = headpose.detect(img_cropped)['angle_y_fc'][0][0]

                #draw facebox on frame
                show_frame = cv2.rectangle(show_frame, (x_min, y_min), (x_max, y_max), color=(0, 255,0), thickness=2)
                show_frame = cv2.putText(show_frame, str(yaw), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, org=(x_min, y_min), color=(0,255,0))
            except Exception as ex:
                print(ex)

    cv2.imshow("aloalo", show_frame)
    cv2.waitKey(1)