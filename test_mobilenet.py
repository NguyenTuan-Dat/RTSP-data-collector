from openvino.inference_engine import IECore
from face_detection import FaceDetection
from glass_mobilenet import GlassMobilenet
import cv2
import numpy as np
import argparse
import time
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", choices=['fmobilenetv3', 'mobilenetv3', 'base_unet'])
parser.add_argument("--cam", "-c", action="store_true")
parser.add_argument("--model_path", "-p", type=str, default="./models/glass_mask_mxnet")
args = parser.parse_args()

INPUT_SHAPE = 48

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

GLASS_MOBILENET_XML_PATH = args.model_path + ".xml"
GLASS_MOBILENET_BIN_PATH = args.model_path + ".bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

glass_detector = GlassMobilenet(ie, GLASS_MOBILENET_XML_PATH, GLASS_MOBILENET_BIN_PATH, input_shape=INPUT_SHAPE)

color = (0, 255, 0)


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


total_time = 0.0
count_time = 0
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

                    if args.model == 'fmobilenetv3':
                        result = glass_detector.detect(img_cropped)['softmax0_softmax0'][0]
                    elif args.model == 'mobilenetv3':
                        result = glass_detector.detect(img_cropped)['mobilenetv30_flatten0_flatten0'][0]
                    else:
                        result = glass_detector.detect(img_cropped)['efficientnet0_output_flatten0_flatten0'][0]
                        print(result)

                    argmax = np.argmax(result)

                    color = [0, 0, 0]
                    color[argmax] = 255
                    color = tuple(color)

                    cv2.putText(frame, text=str(result[argmax]), org=(x_min, y_min),
                                fontFace=cv2.INTER_AREA,
                                fontScale=1,
                                color=color)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)

                except Exception as ex:
                    print(ex)

                total_time += time.time() - t
                count_time += 1

                print("FPS: {}".format(count_time / total_time), end="\r")

        cv2.imshow("AloAlo", frame)
        cv2.waitKey(1)
else:
    # dirs = os.listdir("/Users/ntdat/Downloads/faces-spring-2020")
    # for dir in dirs:
    #     if dir == ".DS_Store":
    #         continue

    for r, d, fs in os.walk("/Users/ntdat/Downloads/mask_224x224/Normal"):
        for f in fs:
            if ".jpg" in f:
                img = cv2.imread(os.path.join(r, f))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = glass_detector.detect(img)['softmax0_softmax0'][0]

                if result[1] > result[0]:
                    shutil.move(os.path.join(r, f),
                                os.path.join("/Users/ntdat/Downloads/mask_224x224/Glass/", f))
                # cv2.putText(img, text=str(result), org=(50, 50), fontFace=cv2.INTER_AREA, fontScale=1,
                #             color=color)
                # cv2.imshow("AloAlo", img)
                # cv2.waitKey()
