import json
import numpy as np
import os
import shutil

FEATURE_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/Cropped_by_scrfd"
NEW_FEATURE_FOLDER = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/Cropped_by_scrfd_new_pose"

if not os.path.exists(NEW_FEATURE_FOLDER):
    os.mkdir(NEW_FEATURE_FOLDER)

headpose_dict = None
with open("/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Data/headpose_.json", "r") as f:
    content = f.read()
    headpose_dict = json.loads(content)

if headpose_dict == None:
    print("headpose_dict is None")

keys = headpose_dict.keys()
print(keys)
count = 0
_from_to = np.zeros((3, 3), dtype=np.int)

for filename in keys:
    arr = filename.split('_')
    if not os.path.exists(os.path.join(NEW_FEATURE_FOLDER, arr[0])):
        os.mkdir(os.path.join(NEW_FEATURE_FOLDER, arr[0]))
    old_pose = int(arr[-2])
    del arr[-2]
    if not os.path.exists(os.path.join(FEATURE_FOLDER, arr[0], filename + ".jpg")):
        print(os.path.join(FEATURE_FOLDER, filename + ".jpg"))
        continue

    yawl = np.abs(float(headpose_dict[filename]["yaw_angle"]))
    change_pose_name = 0 if yawl <= 10 else 1 if yawl <= 30 else 2

    new_filename = arr[0]
    for element in arr[1:-1]:
        new_filename += "_" + element
    new_filename += "_" + str(change_pose_name) + "_" + arr[-1]
    print(filename, new_filename)
    if filename != new_filename:
        count += 1

    _from_to[old_pose][change_pose_name] += 1
    shutil.copyfile(os.path.join(FEATURE_FOLDER, arr[0], filename + ".jpg"),
                    os.path.join(NEW_FEATURE_FOLDER, arr[0], new_filename + ".jpg"))

print("change_name_count: ", count)
print(_from_to)
