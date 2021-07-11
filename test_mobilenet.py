from PIL.ImageFont import ImageFont
from openvino.inference_engine import IECore
from face_detection import FaceDetection
from glass_mobilenet import GlassMobilenet
from HeadposeEstimation import HeadposeEstimation
import cv2
import numpy as np
import argparse
import time
import os
import shutil
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", choices=['fmobilenetv3', 'mobilenetv3', 'base_unet', "multitask"])
parser.add_argument("--cam", "-c", action="store_true")
parser.add_argument("--model_path", "-p", type=str, default="./models/glass_mask_mxnet")
args = parser.parse_args()

INPUT_SHAPE = 48

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

GLASS_MOBILENET_XML_PATH = args.model_path + ".xml"
GLASS_MOBILENET_BIN_PATH = args.model_path + ".bin"

HEADPOSE_XML_PATH = "./models/head-pose-estimation-adas-0001.xml"
HEADPOSE_BIN_PATH = "./models/head-pose-estimation-adas-0001.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

glass_detector = GlassMobilenet(ie, GLASS_MOBILENET_XML_PATH, GLASS_MOBILENET_BIN_PATH, input_shape=INPUT_SHAPE)

headpose = HeadposeEstimation(ie, HEADPOSE_XML_PATH, HEADPOSE_BIN_PATH)

color = (0, 255, 0)


def run_head_pose(face_frame):
    rotate_angle = 0
    yaw_angle = 0
    # run nn
    yaw_angle, pitch_angle, rotate_angle = headpose.detect(face_frame)
    return (True, rotate_angle, yaw_angle)


def correction(frame, center=None, angle=None, invert=False):
    angle = int(angle)
    h, w = frame.shape[:2]
    if center is None:
        center = (h // 2, w // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr


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

                    (status, rotate_angle, yaw_angle) = run_head_pose(img_cropped)

                    img_cropped = correction(img_cropped, angle=rotate_angle)

                    if args.model == 'fmobilenetv3':
                        result = glass_detector.detect(img_cropped)['softmax0_softmax0'][0]
                    elif args.model == 'mobilenetv3':
                        result = glass_detector.detect(img_cropped)['mobilenetv30_flatten0_flatten0'][0]
                    elif args.model == 'multitask':
                        result = glass_detector.detect(img_cropped)
                        glass = np.argmax(result['mobilenetv2multitask0_output_glasses_softmax0_softmax0'][0])
                        mask = np.argmax(result['mobilenetv2multitask0_output_mask_softmax0_softmax0'][0])
                        normal = np.argmax(result['mobilenetv2multitask0_output_normal_softmax0_softmax0'][0])

                        print("detected: {}, {}, {}".format(glass, mask, normal))
                        continue
                    else:
                        result = glass_detector.detect(img_cropped)['mobilenetv20_output_flatten0_flatten0'][0]
                        print(result)

                    argmax = np.argmax(result)

                    color = [0, 0, 0]
                    color[argmax] = 255
                    color = tuple(color)
                    cv2.putText(frame, text="Đạt" + str(result[argmax]), org=(x_min, y_min),
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
    PATH_TO_DATA_FOLDER = "/Users/ntdat/Downloads/face_cropped"
    # PATH_TO_DATA_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/Cropped_by_scrfd_classify"
    classes = ["Glass", "Mask", "Normal"]
    cases = np.zeros((3, 3), dtype=np.int)
    false_name = dict()
    false_name["Glass"] = []
    false_name["Mask"] = []
    false_name["Normal"] = []

    for dir in os.listdir(PATH_TO_DATA_FOLDER):
        if dir != "Glass" and dir != "Mask" and dir != "Normal":
            continue
        for image_name in os.listdir(os.path.join(PATH_TO_DATA_FOLDER, dir)):
            try:
                # if image_name.split("_")[-2] != "0":
                #     continue
                image = cv2.imread(os.path.join(PATH_TO_DATA_FOLDER, dir, image_name))
                result = glass_detector.detect(image)['softmax0_softmax0'][0]
                argmax = np.argmax(result)
                idx = classes.index(dir)
                cases[idx][argmax] += 1
                if argmax != idx:
                    false_name[dir].append({"image_name": image_name, "pred": str(argmax)})
            except Exception as ex:
                print(image_name)

    print("Case", repr(cases))
    with open("False_cases.json", "w") as f:
        f.write(json.dumps(false_name))
