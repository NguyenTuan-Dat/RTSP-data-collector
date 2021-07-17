from openvino.inference_engine import IECore
from human_pose import HumanPose
import cv2
import numpy as np
import argparse
import time
import os

from parse_human_pose.parse_poses import parse_poses

HUMAN_POSE_XML_PATH = "./models/human-pose-estimation-3d-0001.xml"
HUMAN_POSE_BIN_PATH = "./models/human-pose-estimation-3d-0001.bin"

ie = IECore()

human_pose_model = HumanPose(ie, HUMAN_POSE_XML_PATH, HUMAN_POSE_BIN_PATH)

video = cv2.VideoCapture(0)

while video.isOpened():
    _, frame = video.read()

    img = np.zeros((240, 320))

    result = human_pose_model.detect(frame)

    input_scale, stride, fx, is_video = 1, 8, np.float32(0.8 * frame.shape[1]), False

    poses_3d, poses_2d = parse_poses(result, input_scale, stride, fx, is_video)
    print(poses_3d)

    # for idx, feature in enumerate(result["features"][0]):
    #     print(idx, feature)
    # x, y, z = int(x * 320), int(y * 240), int(255 * z)
    # img[x][y] = z

    cv2.imshow("aloalo", img)
    cv2.waitKey(1)
