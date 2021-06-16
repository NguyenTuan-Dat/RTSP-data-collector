from openvino.inference_engine import IECore
from face_detection import FaceDetection
from glass_mobilenet import GlassMobilenet
import cv2
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cam", "-c", action="store_true")
args = parser.parse_args()

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

GLASS_MOBILENET_XML_PATH = "/Volumes/JIOOUM/net_adam_5-0001.xml"
GLASS_MOBILENET_BIN_PATH = "/Volumes/JIOOUM/net_adam_5-0001.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

glass_detector = GlassMobilenet(ie, GLASS_MOBILENET_XML_PATH, GLASS_MOBILENET_BIN_PATH)

color = (0, 255, 0)

if args.cam:
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

                    # img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

                    result = glass_detector.detect(img_cropped)

                    result = result['predictions'][0][0]

                    print(result)

                    if result > 0.6:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(frame, text=str(result), org=(x_min, y_min), fontFace=cv2.INTER_AREA, fontScale=1,
                                color=color)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)

                except Exception as ex:
                    print(ex)

                print("FPS: {}".format(1 / (time.time() - t)), end="\r")

        cv2.imshow("AloAlo", frame)
        cv2.waitKey(1)
else:
    dirs = os.listdir("/Users/ntdat/Downloads/faces-spring-2020")
    for dir in dirs:
        if dir == ".DS_Store":
            continue

        for r, d, fs in os.walk(os.path.join("/Users/ntdat/Downloads/faces-spring-2020", dir)):
            for f in fs:
                if ".png" in f:
                    img = cv2.imread(os.path.join(r, f))
                    result = glass_detector.detect(img)

                    print(result)

                    cv2.putText(img, text=str(result), org=(50, 50), fontFace=cv2.INTER_AREA, fontScale=1,
                                color=color)
                    cv2.imshow("AloAlo", img)
                    cv2.waitKey()
