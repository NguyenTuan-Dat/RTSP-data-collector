import os
import shutil

PATH_TO_DATA = "/Users/ntdat/Downloads/Eufy/Cropped"
PATH_TO_SAVE = "/Users/ntdat/Downloads/Eufy/Cropped"

if not os.path.exists(os.path.join(PATH_TO_SAVE, "Mask")):
    os.mkdir(os.path.join(PATH_TO_SAVE, "Mask"))
if not os.path.exists(os.path.join(PATH_TO_SAVE, "Glass")):
    os.mkdir(os.path.join(PATH_TO_SAVE, "Glass"))
if not os.path.exists(os.path.join(PATH_TO_SAVE, "Normal")):
    os.mkdir(os.path.join(PATH_TO_SAVE, "Normal"))
if not os.path.exists(os.path.join(PATH_TO_SAVE, "Hat")):
    os.mkdir(os.path.join(PATH_TO_SAVE, "Hat"))

# for dir in os.listdir(PATH_TO_DATA):
#     if dir == ".DS_Store":
#         continue
for image_name in os.listdir(PATH_TO_DATA):
    if image_name == ".DS_Store" or image_name == "Glass" or image_name == "Mask" or image_name == "Normal" or image_name == "Hat":
        continue
    if "mask" in image_name:
        shutil.move(os.path.join(PATH_TO_DATA, image_name), os.path.join(PATH_TO_SAVE, "Mask", image_name))
        continue
    elif "glass" in image_name:
        shutil.move(os.path.join(PATH_TO_DATA, image_name), os.path.join(PATH_TO_SAVE, "Glass", image_name))
        continue
    elif "hat" not in image_name:
        shutil.move(os.path.join(PATH_TO_DATA, image_name), os.path.join(PATH_TO_SAVE, "Normal", image_name))
        continue
    elif "hat" in image_name:
        shutil.move(os.path.join(PATH_TO_DATA, image_name), os.path.join(PATH_TO_SAVE, "Hat", image_name))
        continue
