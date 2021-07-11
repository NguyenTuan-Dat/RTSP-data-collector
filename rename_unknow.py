import os

PATH_TO_DATA = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/storage/face_features_unknow_VNCeleb"

for folder in os.listdir(PATH_TO_DATA):
    if folder == "image_dict.json":
        continue
    for idx, file in enumerate(os.listdir(os.path.join(PATH_TO_DATA, folder))):
        os.rename(os.path.join(PATH_TO_DATA, folder, file),
                  os.path.join(PATH_TO_DATA, folder, "unknow_vnceleb_" + file))
