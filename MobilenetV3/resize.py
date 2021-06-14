import cv2
import os
import numpy as np


def read_files(path_to_dir, path_to_save):
    for dir in os.listdir(path_to_dir):
        path_to_folder = os.path.join(path_to_dir, dir)
        os.mkdir(os.path.join(path_to_save, dir))
        if ".DS_Store" == dir:
            continue
        for img_name in os.listdir(path_to_folder):
            if ".DS_Store" == img_name:
                continue
            img = cv2.imread(os.path.join(path_to_folder, img_name))
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(os.path.join(path_to_save, dir, img_name), img)


read_files("/Users/ntdat/Downloads/faces-spring-2020", "/Users/ntdat/Downloads/faces-spring-2020-224x224")
