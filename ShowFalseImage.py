import cv2
import json
import os

PATH_TO_DATA = "/Users/ntdat/Downloads/face_cropped"
# PATH_TO_DATA = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/Cropped_by_scrfd_classify"
PATH_TO_JSON = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/False_cases.json"
PATH_TO_SAVE = "/Users/ntdat/Downloads/False_cases_classify/"
CASE = "Mask"

if not os.path.exists(PATH_TO_SAVE):
    os.mkdir(PATH_TO_SAVE)

content = None
with open(PATH_TO_JSON, "r") as f:
    content_json = f.read()
    content = json.loads(content_json)

for name in content[CASE]:
    try:
        image = cv2.imread(os.path.join(PATH_TO_DATA, CASE, name["image_name"]))
        image = cv2.resize(image, (250, 250))
        cv2.putText(image, name["pred"], (50, 50), color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
        cv2.imwrite(os.path.join(PATH_TO_SAVE, name["image_name"]), image)
        # cv2.imshow("AloALo", image)
        # if cv2.waitKey() == ord("q"):
        #     cv2.destroyAllWindows()
        #     exit()
    except Exception as ex:
        print(ex)
