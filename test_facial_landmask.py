from openvino.inference_engine import IECore
from face_detection import FaceDetection
from facial_landmask import FacialLandmask
import cv2
import numpy as np
import argparse
import time
import os

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

FACIAL_LANDMASK_XML_PATH = "./models/facial-landmarks-35-adas-0002.xml"
FACIAL_LANDMASK_BIN_PATH = "./models/facial-landmarks-35-adas-0002.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

facial_landmask_detector = FacialLandmask(ie, FACIAL_LANDMASK_XML_PATH, FACIAL_LANDMASK_BIN_PATH)

color = (0, 255, 0)


def draw_facial_landmasks(landmasks, face_local, img):
    global color
    x_min, y_min, x_max, y_max = face_local
    x_scales = landmasks[0][0::2]
    y_scales = landmasks[0][1::2]
    points = []
    for idx in range(len(x_scales)):
        x_scale = x_scales[idx]
        y_scale = y_scales[idx]

        x = x_min + int(x_scale * (x_max - x_min))
        y = y_min + int(y_scale * (y_max - y_min))

        if idx >= 20 and idx <= 32 or idx == 4:
            points.append((x, y))

        cv2.circle(img, (x, y), radius=1, color=color)
        cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color)

    points.append(points[0])

    cv2.fillPoly(img, np.int32(np.array([points])), color=(0, 0, 0), lineType=cv2.LINE_AA)


# init video
video = cv2.VideoCapture(0)

while (video.isOpened()):
    _, frame = video.read()
    h, w, c = frame.shape

    # get face on frame
    outputs = facedetection.detect(frame)

    if len(outputs) != 0:
        outputs = np.array(outputs)
        color = (0, 255, 0)
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

                result = facial_landmask_detector.detect(img_cropped)

                draw_facial_landmasks(result, (x_min, y_min, x_max, y_max), frame)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)

            except Exception as ex:
                print(ex)

            print("FPS: {}".format(1 / (time.time() - t)), end="\r")

    cv2.imshow("AloAlo", frame)
    cv2.waitKey(1)
