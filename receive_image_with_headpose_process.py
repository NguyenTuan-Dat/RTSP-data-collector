from openvino.inference_engine import IECore
from head_pose import HeadPose
from face_detection import FaceDetection
import cv2
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-name', '--name', type=str)
parser.add_argument('-ipp', '--img_per_pose', type=int)
parser.add_argument('-g', '--glass', action="store_true")
parser.add_argument('-hh', '--hat', action="store_true")
parser.add_argument('-m', '--mask', action="store_true")
parser.add_argument('-b', '--brightness', action="store_true")
parser.add_argument('-d', '--debug', action="store_true")

args = parser.parse_args()


#path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

HEADPOSE_XML_PATH = "./models/head-pose-estimation-adas-0001.xml"
HEADPOSE_BIN_PATH = "./models/head-pose-estimation-adas-0001.bin"


#Cam rtsp link
CAM_ADDR = "/Users/ntdat/Downloads/Data_Collected_by_EufyCamera/47_mask_0.mp4"


#Data dir
DATA_DIR = "data_collection"


# 0-10, 10-30, 30-45
total_saved_img = [0, 0, 0]
img_per_pose = args.img_per_pose

def save_image_by_yawl(image, file_name, yawl):
    # solve save data

    if 0 <= yawl < 10 and total_saved_img[0] < img_per_pose:
        file_name = DATA_DIR + "/" + file_name + "_0_" + str(time.time()) + ".jpg"
        total_saved_img[0] += 1
        cv2.imwrite(file_name, image)
    elif 10 <= yawl < 30 and total_saved_img[1] < img_per_pose:
        file_name = DATA_DIR + "/" + file_name + "_1_" + str(time.time()) + ".jpg"
        total_saved_img[1] += 1
        cv2.imwrite(file_name, image)
    elif 30 <= yawl < 45 and total_saved_img[2] < img_per_pose:
        file_name = DATA_DIR + "/" + file_name + "_2_" + str(time.time()) + ".jpg"
        total_saved_img[2] += 1
        cv2.imwrite(file_name, image)

    if ".jpg" in file_name:
        print("saved img: {}".format(file_name))
    else:
        print("This yaw is done!!!!!!!!")


ie = IECore()

#Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

#Create HeadPose model
headpose = HeadPose(ie, HEADPOSE_XML_PATH, HEADPOSE_BIN_PATH)

#init video from rtsp link
video = cv2.VideoCapture(CAM_ADDR)

#init video from build-in camera
# video = cv2.VideoCapture(0)


case = ["hat", "glass", "mask", "brightness"]

if not args.hat:
    case.remove("hat")
if not args.glass:
    case.remove("glass")
if not args.mask:
    case.remove("mask")
if not args.brightness:
    case.remove("brightness")

try:
    os.makedirs(os.path.join(DATA_DIR, args.name))
except:
    pass

debug = args.debug

count = 0

while(video.isOpened()):
    count += 1
    if count < 200:
        continue
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
                if debug:
                    #draw facebox on frame
                    show_frame = cv2.rectangle(show_frame, (x_min, y_min), (x_max, y_max), color=(0, 255,0), thickness=2)
                    show_frame = cv2.putText(show_frame, str(yaw), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, org=(x_min, y_min), color=(0,255,0))
                    cv2.imshow("aloalo", show_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('h'):
                        if "hat" in case:
                            case.remove("hat")
                        else:
                            case.append("hat")
                    elif key == ord('g'):
                        if "glass" in case:
                            case.remove("glass")
                        else:
                            case.append("glass")
                    elif key == ord('m'):
                        if "mask" in case:
                            case.remove("mask")
                        else:
                            case.append("mask")
                    elif key == ord('b'):
                        if "brightness" in case:
                            case.remove("brightness")
                        else:
                            case.append("brightness")
                    elif key == ord('d'):
                        isDone = not isDone
                        if not isDone:
                            total_saved_img = [1, 1, 1]

                file_name = "_".join([
                    args.name,
                    "hat" if "hat" in case else "0",
                    "glass" if "glass" in case else "0",
                    "mask" if "mask" in case else "0",
                    "brightness" if "brightness" in case else "0",
                ])

                file_name = os.path.join(args.name, file_name)

                save_image_by_yawl(frame, file_name, abs(yaw))
            except Exception as ex:
                print(ex)

