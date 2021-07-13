import cv2
import time
import os

ROOT_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/TestAutoSave"

CASE = [
    [
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 0_1626171995.31201.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 0_1626172189.075299.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 0_1626172085.05697.png"
    ],
    [
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 1_1626172484.980748.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 1_1626172375.781922.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 1_1626172549.662718.png"
    ],
    [
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 2_1626172753.039148.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 2_1626172681.978404.png",
        "db_model_mask_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 2_1626172615.4936.png"
    ]

]

imgs = []

for case in range(3):
    case_imgs = []
    for pose in range(3):
        path = os.path.join(ROOT_FOLDER, CASE[case][pose])
        img = cv2.imread(path)
        case_imgs.append(img)
    concat_img = cv2.hconcat(case_imgs)
    imgs.append(concat_img)

cv2.imwrite(
    "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/arcface_model/Model_mask_unknow_vnceleb_all_pose.png",
    cv2.vconcat(imgs))
