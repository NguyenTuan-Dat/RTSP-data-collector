import os
import time

import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import sys
import argparse
from numpy import arange
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--unknow", action="store_true")
parser.add_argument("--unknow_vnceleb", action="store_true")
args = parser.parse_args()

FEATURES_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/features/face_features_arcface_affine"
DB_FILE = '/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/features/face_features_HN'
SAVE_PLOT_PATH = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/TestAutoSave/"
TAR_AND_FAR = ""

fig_names = list()

TOTAL_OF_NON_MATE_SEARCHES = 0
TOTAL_OF_MATE_SEARCHES = 0

NUM_OF_NON_MATE_SEARCHES = 0
NUM_OF_MATE_SEARCHES = 0


def load_db(path):
    db_dic = dict()
    for npy in read_feature_folder(path):
        real_user_id, file_name, test_case = split_filename_infor(npy)


def filename2testcase(file_name):
    """
    filename format:
        <id>_<glass>_<hat>_<mask>_<brightness>_<angle_range>_<time>
    """

    PREFIX_2_TESTCASE = {
        "_0_0_0_0_0_": 22,
        "_0_0_0_0_1_": 23,
        "_0_0_0_0_2_": 24,
        "_0_0_0_brightness_0_": 19,
        "_0_0_0_brightness_1_": 20,
        "_0_0_0_brightness_2_": 21,
        "_0_0_mask_0_0_": 12,
        "_0_0_mask_0_1_": 15,
        "_0_0_mask_0_2_": 18,
        "_0_0_mask_brightness_0_": 3,
        "_0_0_mask_brightness_1_": 6,
        "_0_0_mask_brightness_2_": 9,
        "_0_glass_0_0_0_": 10,
        "_0_glass_0_0_1_": 13,
        "_0_glass_0_0_2_": 16,
        "_0_glass_0_brightness_0_": 1,
        "_0_glass_0_brightness_1_": 4,
        "_0_glass_0_brightness_2_": 7,
        "_hat_0_0_0_0_": 11,
        "_hat_0_0_0_1_": 14,
        "_hat_0_0_0_2_": 17,
        "_hat_0_0_brightness_0_": 2,
        "_hat_0_0_brightness_1_": 5,
        "_hat_0_0_brightness_2_": 8,
        "_vnceleb_": -2,
        "unknow_": -1,
    }

    for prefix in PREFIX_2_TESTCASE.keys():
        if prefix in file_name:
            return PREFIX_2_TESTCASE[prefix]
    return None


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    return (similarity + 1) / 2


def get_min_distance(face_embedding, face_types=("normal", "glasses", "mask",), metric="cosine"):
    if face_embedding is None:
        raise ValueError("face_embedding can not be None!")
    else:
        max_ = -99999
        label_ = -1

        for label in list(labels):
            if face_types is None:
                face_types = db_dic[label].keys()

            for type in face_types:
                if type not in db_dic[label].keys():
                    print("Not found {} in database of {}".format(type, label))
                    continue
                for j in db_dic[label][type]:
                    conf_ = 0
                    if metric == "cosine":
                        conf_ = cosine_distance(j, face_embedding)
                    else:
                        raise ValueError("Metric not supported")

                    if conf_ > max_:
                        max_ = conf_
                        label_ = label
        return (max_, label_)


def split_filename_infor(file_path):
    if file_path is None:
        raise ValueError("Please input file_path")

    root = os.path.split(file_path)[0]
    file_name = os.path.splitext(os.path.split(file_path)[1])[0]
    user_id = os.path.split(root)[-1]
    test_case = filename2testcase(file_name)

    return user_id, file_name, test_case


def read_feature_folder(feature_folder):
    npy_paths = []
    for user in os.listdir(feature_folder):
        if user == ".DS_Store" or user == "image_dict.json":
            continue
        for npy in os.listdir(os.path.join(feature_folder, user)):
            if npy == ".DS_Store":
                continue
            npy_paths.append(os.path.join(feature_folder, user, npy))
    return npy_paths


def calculate_accuracy(feature_folder, face_types=("normal", "glasses", "mask"), using_headpose=(0, 1, 2),
                       threshold=(0.35, 0.55)):
    global NUM_OF_MATE_SEARCHES, NUM_OF_NON_MATE_SEARCHES, TAR_AND_FAR, TOTAL_OF_MATE_SEARCHES, TOTAL_OF_NON_MATE_SEARCHES

    using_testcase = find_using_testcase(face_types, using_headpose)

    # True Possitive
    TP = 0
    # True Negative
    TN = 0
    # False Possitive - Predict is user_1 but real label is unknown or other user
    FP = 0
    # False Negative
    FN = 0
    # Number of total images
    n_images = 0
    # Total
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    true_distances = []
    false_distances = []

    list_labels = list(labels)

    for feature_path in read_feature_folder(feature_folder):
        try:
            real_user_id, file_name, test_case = split_filename_infor(feature_path)
            if test_case is not None and test_case not in using_testcase:
                continue

            # set verify image
            # get ebedding
            # print(FEATURES_FOLDER, real_user_id, file_name + ".npy")

            if real_user_id in db_dic.keys():
                NUM_OF_MATE_SEARCHES += 1
                TOTAL_OF_MATE_SEARCHES += 1
            else:
                NUM_OF_NON_MATE_SEARCHES += 1
                TOTAL_OF_NON_MATE_SEARCHES += 1

            face_embedding = np.load(os.path.join(FEATURES_FOLDER, real_user_id, file_name + ".npy"))

            face_type = ("normal",)
            if test_case in [12, 15, 18, 3, 6, 9]:
                # face_type = ("mask",)
                face_type = ("normal",)
            if test_case in [10, 13, 16, 1, 4, 7]:
                face_type = ("glasses",)

            (predict_distance, predict_user_id) = get_min_distance(
                face_embedding,
                metric="cosine",
                face_types=face_type)

            if real_user_id in labels and real_user_id == predict_user_id:
                true_distances.append(predict_distance * 100)
            if real_user_id not in labels or real_user_id != predict_user_id:
                false_distances.append(predict_distance * 100)

            predict_user_id = str(predict_user_id)

            if predict_distance >= threshold[0] and predict_distance < threshold[1]:
                if real_user_id in labels:
                    NUM_OF_MATE_SEARCHES -= 1
                else:
                    NUM_OF_NON_MATE_SEARCHES -= 1
                continue

            log_message = "File name: " + file_name + " Predict user_id: " + predict_user_id + " Predict distance: " + str(
                predict_distance)

            # True Possitive
            if predict_user_id == real_user_id and predict_distance >= threshold[1]:
                log_message += " is TP"
                total_TP += predict_distance
                TP += 1

            # True Negative
            if real_user_id not in labels and predict_distance < threshold[1]:
                total_TN += predict_distance
                log_message += " is TN"
                TN += 1

            # False Possitive - Predict is user_1 but real label is unknown or other user
            if (real_user_id not in labels or real_user_id != predict_user_id) \
                    and predict_distance >= threshold[1]:
                total_FP += predict_distance
                log_message += " is FP"
                FP += 1

            # False Negative
            if real_user_id in labels and predict_distance < threshold[1]:
                total_FN += predict_distance
                log_message += " is FN"
                FN += 1

            with open('accuracy_logs', 'a') as f:
                f.writelines([log_message + "\n"])

            n_images += 1
            # TAR_AND_FAR = "TP:\t{} TN:\t{} FP:\t{} FN:\t{} Total:\t{} TAR@FAR: {:.4f}@{:.4f} FPIR: {:.4f} FNIR: {:.4f} TPIR: {:.4f} TNIR: {:.4f} Threshold: {:.2f}->{:.2f}".format(
            #     TP,
            #     TN,
            #     FP,
            #     FN,
            #     n_images,
            #     TP / (TP + FN),
            #     FP / (FP + TN),
            #     FP / NUM_OF_NON_MATE_SEARCHES,
            #     FN / NUM_OF_MATE_SEARCHES,
            #     TP / NUM_OF_MATE_SEARCHES,
            #     TN / NUM_OF_NON_MATE_SEARCHES,
            #     threshold[0],
            #     threshold[1]
            # )
            # print(feature_path)
            # print(TAR_AND_FAR)
        except Exception as ex:
            print(ex)
            continue
    fig_name = draw_plot(true_distances, false_distances, face_types, using_headpose)
    with open(SAVE_PLOT_PATH + "log.txt", "a") as f:
        f.write(fig_name + " " + TAR_AND_FAR + "\n")
    table = [
        int(TP),
        int(TN),
        int(FP),
        int(FN),
        int(NUM_OF_MATE_SEARCHES),
        int(NUM_OF_NON_MATE_SEARCHES),
        FP / NUM_OF_NON_MATE_SEARCHES,
        # FN / NUM_OF_MATE_SEARCHES,
        # TP / NUM_OF_MATE_SEARCHES,
        FN / 1,
        TP / 1,
        TN / NUM_OF_NON_MATE_SEARCHES,
        int(TOTAL_OF_MATE_SEARCHES),
        int(TOTAL_OF_NON_MATE_SEARCHES),
    ]

    print("Done", face_types)

    return table, fig_name


def find_using_testcase(face_types=("normal", "glasses", "mask"), using_headpose=(0, 1, 2)):
    face_types_dict = dict()
    face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
    face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
    face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
    face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

    headpose_dict = [
        [22, 19, 12, 3, 10, 1, 11, 2],  # rotate angle: abs <= 10
        [23, 20, 15, 6, 13, 4, 14, 5],  # rotate angle: 10 < abs <= 30
        [24, 21, 18, 9, 16, 7, 17, 8],  # rotate angle: 30 < abs <=45
    ]

    face_type_testcases = []
    for face_type in face_types:
        face_type_testcases.extend(face_types_dict[face_type])

    headpose_testcase = []
    for headpose in using_headpose:
        headpose_testcase.extend(headpose_dict[headpose])

    if args.unknow_vnceleb:
        face_type_testcases.extend([-2])
        headpose_testcase.extend([-2])
    if args.unknow:
        face_type_testcases.extend([-1])
        headpose_testcase.extend([-1])

    using_testcase = np.intersect1d(face_type_testcases, headpose_testcase)

    print(using_testcase)
    return using_testcase


def draw_distibution(feature_folder, face_types=("normal", "glasses", "mask"), using_headpose=(0, 1, 2)):
    using_testcase = find_using_testcase(face_types, using_headpose)
    true_distances = []
    false_distances = []
    list_labels = list(labels)
    for feature_path in read_feature_folder(feature_folder):
        real_user_id, file_name, test_case = split_filename_infor(feature_path)

        if test_case not in using_testcase:
            continue
        run_status, distance, _ = get_distance_of_user(real_user_id, file_name, test_case, face_types, list_labels)
        if run_status:
            true_distances.append(distance[1] * 100)
        false_distances.append(distance[0] * 100)

    draw_plot(true_distances, false_distances, face_types, using_headpose)


def get_distance_of_user(real_user_id, file_name, test_case, face_types, list_labels):
    false_id = np.random.randint(0, len(labels))
    while list_labels[false_id] == str(real_user_id):
        false_id = np.random.randint(0, len(list_labels))

    face_embedding = np.load(os.path.join(FEATURES_FOLDER, real_user_id, file_name + ".npy"))

    false_distance = get_max_conf(face_embedding, list_labels[false_id], face_types=face_types)

    if str(real_user_id) in labels:
        true_distance = get_max_conf(face_embedding, str(real_user_id), face_types=face_types)
        return True, (false_distance, true_distance), list_labels[false_id]
    return False, (false_distance,), list_labels[false_id]


def get_max_conf(face_embedding, user_id, face_types=("normal", "glasses", "mask")):
    max_distance = -999
    for type in face_types:
        for idx in db_dic[user_id][type]:
            distance = cosine_distance(face_embedding, idx)
            if distance > max_distance:
                max_distance = distance

    return max_distance


def draw_all_case(feature_folder, face_types=("normal", "glasses", "mask"), using_headpose=(0, 1, 2)):
    using_testcase = find_using_testcase(face_types, using_headpose)
    true_distances = []
    false_distances = []
    list_labels = list(labels)
    for feature_path in read_feature_folder(feature_folder):
        real_user_id, file_name, test_case = split_filename_infor(feature_path)

        if test_case not in using_testcase:
            continue

        face_embedding = np.load(
            os.path.join(FEATURES_FOLDER, real_user_id, file_name + ".npy"))

        for user_id in labels:
            distance = get_max_conf(face_embedding, user_id, face_types=face_types) * 100

            if (user_id == real_user_id):
                true_distances.append(distance)
            else:
                false_distances.append(distance)

    draw_plot(true_distances, false_distances, face_types, using_headpose)


def draw_plot(true_distances, false_distances, face_types, using_headpose, bins=arange(0, 100, 0.5)):
    title_cases = ""
    for idx, case in enumerate(face_types):
        title_cases += case.capitalize()
        if idx != len(face_types) - 1:
            title_cases += ", "

    title_cases += " on headpose: "
    for idx, case in enumerate(using_headpose):
        title_cases += str(case)
        if idx != len(using_headpose) - 1:
            title_cases += ", "

    plt.title("Distributions of " + title_cases)
    plt.hist(true_distances, alpha=0.6, color="g", bins=bins, )
    plt.hist(false_distances, alpha=0.6, color="r", bins=bins)
    fig_name = DB_FILE.split("/")[-1].split(".")[0] + \
               "_" + \
               FEATURES_FOLDER.split("/")[-1] + \
               "_" + \
               title_cases.split(":")[0] + \
               title_cases.split(":")[1] + \
               "_" + \
               str(time.time()) + \
               ".png"

    plt.savefig(os.path.join(SAVE_PLOT_PATH + fig_name))
    # plt.show()
    plt.clf()

    return fig_name


# print(labels)
# face_types = ("glasses", "mask", "normal")
# using_headposes = (0, 1, 2)
# tables = []
#
# for using_headpose in using_headposes:
#     fig_face_types = list()
#     for face_type in face_types:
#         TOTAL_OF_NON_MATE_SEARCHES = 0
#         TOTAL_OF_MATE_SEARCHES = 0
#
#         NUM_OF_NON_MATE_SEARCHES = 0
#         NUM_OF_MATE_SEARCHES = 0
#         threshold = None
#         _face_type = (face_type,)
#         _using_headpose = (using_headpose,)
#
#         print(_face_type, _using_headpose)
#
#         if "glasses" in face_type:
#             threshold = (0.65, 0.75)
#
#         if "mask" in face_type:
#             threshold = (0.6, 0.75)
#
#         if "normal" in face_type:
#             threshold = (0.7, 0.75)
#
#         table, fig_name = calculate_accuracy(feature_folder=FEATURES_FOLDER, face_types=_face_type,
#                                              using_headpose=_using_headpose, threshold=threshold)
#
#         fig_face_types.append(fig_name)
#
#         print("T_OF_MATE_SEARCHES", TOTAL_OF_MATE_SEARCHES)
#         print("T_OF_NON_MATE_SEARCHES", TOTAL_OF_NON_MATE_SEARCHES)
#
#         tables.append(table)
#
#         print(pd.DataFrame(tables))
#     fig_names.append(fig_face_types)
# print(fig_names)

# draw_distibution(feature_folder=FEATURES_FOLDER, face_types=face_types,
#                  using_headpose=using_headpose)
# draw_all_case(feature_folder="storage/face_features", face_types=face_types,
#               using_headpose=using_headpose)

load_db(DB_FILE)
