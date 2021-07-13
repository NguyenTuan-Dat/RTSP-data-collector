import os

import numpy as np
import pandas as pd

FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/storage/face_features_arcface_affine/"

CASE = {
    "_0_glass_0_0_0_": 0,
    "_0_glass_0_0_1_": 1,
    "_0_glass_0_0_2_": 2,
    "_0_glass_0_brightness_0_": 0,
    "_0_glass_0_brightness_1_": 1,
    "_0_glass_0_brightness_2_": 2,

    "_0_0_mask_0_0_": 10,
    "_0_0_mask_0_1_": 11,
    "_0_0_mask_0_2_": 12,
    "_0_0_mask_brightness_0_": 10,
    "_0_0_mask_brightness_1_": 11,
    "_0_0_mask_brightness_2_": 12,

    "_0_0_0_0_0_": 20,
    "_0_0_0_0_1_": 21,
    "_0_0_0_0_2_": 22,
    "_0_0_0_brightness_0_": 20,
    "_0_0_0_brightness_1_": 21,
    "_0_0_0_brightness_2_": 22,

    "_hat_0_0_0_0_": 30,
    "_hat_0_0_0_1_": 31,
    "_hat_0_0_0_2_": 32,
    "_hat_0_0_brightness_0_": 30,
    "_hat_0_0_brightness_1_": 31,
    "_hat_0_0_brightness_2_": 32,
}


def filename2case(file_name):
    for case in CASE.keys():
        if case in file_name:
            return CASE[case]


count = np.zeros((4, 3), dtype=np.uint16)
for dir in os.listdir(FOLDER):
    try:
        folder = os.path.join(FOLDER, dir)
        # folder = FOLDER
        for file_name in os.listdir(folder):
            case = filename2case(file_name)
            count[case // 10][case % 10] += 1
    except Exception as ex:
        print(ex)
print(pd.DataFrame(count))
