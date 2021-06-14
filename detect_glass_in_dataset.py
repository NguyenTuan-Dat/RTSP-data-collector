import os
from face_detection import FaceDetection
import dlib
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import shutil

PATH_TO_DATA = "/Users/ntdat/Downloads/"
OUR_FACE_DETECTION = True


def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


# ==============================================================================
#   2.绘制回归线 & 找瞳孔函数
#       输入：图片 & numpy格式的landmarks
#       输出：左瞳孔坐标 & 右瞳孔坐标
# ==============================================================================
def get_centers(img, landmarks):
    # 线性回归
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0] + EYE_LEFT_INNER[0]) / 2
    x_right = (EYE_RIGHT_OUTTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)  # 画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


# ==============================================================================
#   3.人脸对齐函数
#       输入：图片 & 左瞳孔坐标 & 右瞳孔坐标
#       输出：对齐后的人脸图片
# ==============================================================================
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)  # 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    cv2.imshow('sobel_y', sobel_y)

    edgeness = sobel_y

    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1 * 0.3 + measure_2 * 0.7

    cv2.imshow('roi_1', roi_1)
    cv2.imshow('roi_2', roi_2)

    if measure > 0.15:
        judge = True
    else:
        judge = False
    return judge


# path to models
predictor_path = "./models/shape_predictor_5_face_landmarks.dat"

FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)
predictor = dlib.shape_predictor(predictor_path)

detector = dlib.get_frontal_face_detector()  # 人脸检测器detector

dirs = os.listdir(PATH_TO_DATA)

for dir in dirs:
    count_true = 0

    PATH_TO_DIR = os.path.join(PATH_TO_DATA, dir)
    PATH_TO_GLASS_DIR = os.path.join(PATH_TO_DIR, "Glass")
    imgs = None

    if dir != "faces-spring-2020":
        continue

    try:
        imgs = os.listdir(PATH_TO_DIR)
        os.mkdir(PATH_TO_GLASS_DIR)
    except Exception as ex:
        print(ex)

    if imgs is None:
        continue

    for img_name in imgs:
        img = cv2.imread(os.path.join(PATH_TO_DIR, img_name))
        try:
            img = cv2.resize(img, (500, 500))
            h, w, c = img.shape
        except:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = []

        if OUR_FACE_DETECTION:
            outputs = facedetection.detect(img)

            i = 0
            for output in outputs:
                x_min, y_min, x_max, y_max = (output * [w, h, w, h]).astype(int)

                if x_max - x_min < y_max - y_min:
                    y_max += ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min -= ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min = 0 if y_min < 0 else int(y_min)
                    y_max = h if y_max > h else int(y_max)
                else:
                    x_max += ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min -= ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min = 0 if x_min < 0 else int(x_min)
                    x_max = w if x_max > w else int(x_max)

                x_face = x_min
                y_face = y_min
                w_face = x_max - x_min
                h_face = y_max - y_min

                rect = dlib.rectangle(x_min, y_min, x_max, y_max)
                rects.append(rect)
        else:
            # dlib detector========================================================#
            rects = detector(gray, 1)
            # =================================================================#

        for i, rect in enumerate(rects):
            # 得到坐标
            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face
            cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
            cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            i += 1

            landmarks = predictor(gray, rect)
            landmarks = landmarks_to_np(landmarks)
            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

            LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

            aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
            cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

            judge = judge_eyeglass(aligned_face)
            if judge == True:
                shutil.move(os.path.join(PATH_TO_DIR, img_name), os.path.join(PATH_TO_GLASS_DIR, img_name))
                cv2.putText(img, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(img, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2,
                            cv2.LINE_AA)

        cv2.imshow("Result", img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    print("count_true in {}: {}".format(dir, count_true))

cv2.destroyAllWindows()
