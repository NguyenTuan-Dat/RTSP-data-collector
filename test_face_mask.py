from openvino.inference_engine import IECore
from face_detection import FaceDetection
from face_mask_classify import FaceMaskClassify
import cv2
import numpy as np
import argparse
import time
import os

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

FACEMASKCLASSIFY_XML_PATH = "./models/sbd_mask.xml"
FACEMASKCLASSIFY_BIN_PATH = "./models/sbd_mask.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

facemask_classify = FaceMaskClassify(ie, FACEMASKCLASSIFY_XML_PATH, FACEMASKCLASSIFY_BIN_PATH)

# init video
video = cv2.VideoCapture(0)

while (video.isOpened()):
    _, frame = video.read()
    h, w, c = frame.shape

    # get face on frame
    outputs = facedetection.detect(frame)

    if len(outputs) != 0:
        outputs = np.array(outputs)
        for output in outputs:
            t = time.time()
            try:
                # get face location
                x_min, y_min, x_max, y_max = (output * [w, h, w, h]).astype(int)

                if x_max - x_min > y_max - y_min:
                    y_max += ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min -= ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min = 0 if y_min < 0 else int(y_min)
                    y_max = h if y_max > h else int(y_max)
                else:
                    x_max += ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min -= ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min = 0 if x_min < 0 else int(x_min)
                    x_max = w if x_max > w else int(x_max)

                # crop face
                img_cropped = frame[y_min:y_max, x_min:x_max]

                img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

                result = facemask_classify.classify(img_cropped)

                print(result)

                color = (0, 255, 0)
                value = result['349'][0][0]
                if result['349'][0][0] > 2:
                    color = (0, 0, 255)
                    value = result['349'][0][0]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)
                cv2.putText(frame, str(np.array(value)), (x_min, y_min), color=color, fontScale=1.0,
                            fontFace=cv2.CALIB_FIX_K1)

            except Exception as ex:
                print(ex)

            print("FPS: {}".format(1 / (time.time() - t)), end="\r")

    cv2.imshow("AloAlo", frame)
    cv2.waitKey(1)
