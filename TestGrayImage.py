import cv2
import os
import numpy as np

PATH_TO_DATA = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/faces-spring-2020-224_mxnet/train/Glass"

for img_name in os.listdir(PATH_TO_DATA):
    if img_name == ".DS_Store":
        continue

    img = cv2.imread(os.path.join(PATH_TO_DATA, img_name))
    img = cv2.imread(
        "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/faces-spring-2020-224_mxnet/train/Glass/73621375@N00_identity_0@7353575_0.jpg")
    print(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    if np.array_equal(r, g) and np.array_equal(r, b):
        cv2.imshow("aloalo", img)
        cv2.waitKey()
        os.remove(os.path.join(PATH_TO_DATA, img_name))
