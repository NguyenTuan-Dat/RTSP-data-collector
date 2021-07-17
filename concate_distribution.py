import cv2
import time
import os

ROOT_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/TestAutoSave"

CASE = [
    [
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 0_1626232625.8893461.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 0_1626232860.093028.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 0_1626233870.7573419.png"
    ],
    [
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 1_1626233898.9690828.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 1_1626233933.0820138.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 1_1626233970.656728.png"
    ],
    [
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Glasses on headpose 2_1626234006.0665028.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Mask on headpose 2_1626234036.617314.png",
        "db_model_mask_non_glasses_affine_face_features_model_mask_new_pose_Normal on headpose 2_1626234078.185689.png"
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
    "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/arcface_model/Model_mask_unknow_vnceleb_all_pose_db2.png",
    cv2.vconcat(imgs))
