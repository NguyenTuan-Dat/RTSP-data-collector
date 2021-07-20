import os
import shutil
import cv2

IMG_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/Cropped_by_scrfd_new_pose"
SAVE_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/DB_HN_Failure_cases/Model_1"

accuracy_logs_content = ""

with open("/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/accuracy_logs_Arcface_HN", "r") as f:
    accuracy_logs_content = f.read()

count_unsure_true_mate = 0
count_unsure_false_mate = 0
count_unsure_unknow = 0

for idx, line in enumerate(accuracy_logs_content.split("\n")):
    try:
        threshold = (0.7, 0.75)
        type = "Normal"
        file_name, distance = line.split(" Predict user_id: ")
        pred_id = distance.split(" ")[0]
        file_name = file_name.split(": ")[-1]
        real_id = file_name.split("_")[0]
        if real_id == "unknow":
            real_id = -1
        else:
            real_id = int(real_id)
        distance = float(distance.split(": ")[-1].split(" is ")[0])
        if "mask" in file_name:
            threshold = (0.6, 0.75)
        type = "Mask"
        if "glass" in file_name:
            threshold = (0.65, 0.75)
        type = "Glass"

        # if distance >= threshold[0] and distance < threshold[1]:
        #     path_to_save_folder = os.path.join(SAVE_FOLDER, type)
        #
        #     if real_id == int(pred_id):
        #         count_unsure_true_mate += 1
        #     elif real_id == -1:
        #         count_unsure_false_mate += 1
        #     else:
        #         count_unsure_unknow += 1

        if distance >= threshold[1]:
            path_to_save_folder = os.path.join(SAVE_FOLDER, type)

            if not os.path.exists(path_to_save_folder):
                os.mkdir(path_to_save_folder)
            print(os.path.join(IMG_FOLDER, file_name.split("_")[0], file_name + ".jpg"))
            # img = cv2.imread(
            #     os.path.join(IMG_FOLDER, file_name.split("_")[2], file_name.split("vnceleb_")[-1] + ".jpg"))
            img = cv2.imread(os.path.join(IMG_FOLDER, file_name.split("_")[0], file_name + ".jpg"))
            print(img)
            img = cv2.resize(img, (224, 224))
            cv2.putText(img, pred_id, (20, 20), fontFace=cv2.FONT_ITALIC, fontScale=0.5,
                        color=(0, 255, 0))
            cv2.putText(img, str(distance), (20, 40), fontFace=cv2.FONT_ITALIC, fontScale=0.5,
                        color=(0, 255, 0))

            cv2.imwrite(os.path.join(path_to_save_folder, file_name + ".jpg"), img)
            # try:
            #     shutil.copy(os.path.join(IMG_FOLDER, file_name.split("_")[0], file_name + ".jpg"),
            #                 os.path.join(path_to_save_folder, file_name + ".jpg"))
            # except Exception as ex:
            #     print(ex)
            #     print(idx, file_name, distance)
    except:
        pass
print(count_unsure_true_mate, count_unsure_false_mate, count_unsure_unknow)
