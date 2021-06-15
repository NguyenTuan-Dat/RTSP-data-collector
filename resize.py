import cv2
import os
import numpy as np
from face_detection import FaceDetection
from openvino.inference_engine import IECore

PATH_TO_SAVE = "/Users/ntdat/Downloads/faces-spring-2020-224x224/Normal/"

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

GLASS_MOBILENET_XML_PATH = "/Volumes/JIOOUM/glass_tf_4000.xml"
GLASS_MOBILENET_BIN_PATH = "/Volumes/JIOOUM/glass_tf_4000.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)


# def read_files(path_to_dir):
#     for img_name in os.listdir(path_to_dir):
#         if ".DS_Store" == img_name:
#             continue
#
#         img = cv2.imread(os.path.join(path_to_dir, img_name))
#         h, w, c = img.shape
#
#         # get face on frame
#         outputs = facedetection.detect(img)
#
#         if len(outputs) != 0:
#             outputs = np.array(outputs)
#             color = (0, 255, 0)
#             for output in outputs:
#                 try:
#                     # get face location
#                     x_min, y_min, x_max, y_max = (output * [w, h, w, h]).astype(int)
#
#                     if x_max - x_min > y_max - y_min:
#                         y_max += ((x_max - x_min) - (y_max - y_min)) / 2
#                         y_min -= ((x_max - x_min) - (y_max - y_min)) / 2
#                         y_min = 0 if y_min < 0 else int(y_min)
#                         y_max = h if y_max > h else int(y_max)
#                     else:
#                         x_max += ((y_max - y_min) - (x_max - x_min)) / 2
#                         x_min -= ((y_max - y_min) - (x_max - x_min)) / 2
#                         x_min = 0 if x_min < 0 else int(x_min)
#                         x_max = w if x_max > w else int(x_max)
#
#                     # crop face
#                     img_cropped = img[y_min:y_max, x_min:x_max]
#                     img_cropped = cv2.resize(img_cropped, (224, 224))
#                     cv2.imwrite(os.path.join(PATH_TO_SAVE, img_name), img_cropped)
#                 except Exception as ex:
#                     print(ex)


def read_files(path_to_folder):
    dirs = os.listdir(path_to_folder)

    for dir in dirs:
        if dir == ".DS_Store":
            continue
        # if os.path.exists(os.path.join(path_to_folder, dir, "Glass")):
        for image in os.listdir(os.path.join(path_to_folder, dir)):
            if image == ".DS_Store":
                continue
            try:
                img = cv2.imread(os.path.join(path_to_folder, dir, image))
                cv2.imshow("aloalo", img)
                key = cv2.waitKey()
                if key == ord("s"):
                    img = cv2.resize(img, (224, 224))
                    cv2.imwrite(os.path.join(PATH_TO_SAVE, dir + image), img)
            except Exception as ex:
                print(ex)


read_files("/Users/ntdat/Downloads/Data_collected_on_insta/instagrams/")
