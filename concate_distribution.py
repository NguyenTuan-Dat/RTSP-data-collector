import cv2
import time
import os

ROOT_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/TestAutoSave"

CASE = [['db_model_arcface_HN_2_face_features_arcface_affine_Glasses on headpose 0_1626523921.075588.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Mask on headpose 0_1626523957.4806132.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Normal on headpose 0_1626523996.4770958.png'],
        ['db_model_arcface_HN_2_face_features_arcface_affine_Glasses on headpose 1_1626524047.2216141.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Mask on headpose 1_1626524091.384284.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Normal on headpose 1_1626524140.724799.png'],
        ['db_model_arcface_HN_2_face_features_arcface_affine_Glasses on headpose 2_1626524196.60527.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Mask on headpose 2_1626524241.769824.png',
         'db_model_arcface_HN_2_face_features_arcface_affine_Normal on headpose 2_1626524292.537808.png']]

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
    "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/distribution_reruned_headpose/arcface_model/Model_arcface_unknow_vnceleb_all_pose_dbHN.png",
    cv2.vconcat(imgs))
