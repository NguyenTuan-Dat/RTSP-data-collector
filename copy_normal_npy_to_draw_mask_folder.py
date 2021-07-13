import os
import shutil

FROM = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/storage/face_features_arcface_affine"
TO = "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/fid-face/storage/face_features_arcface_affine_draw_mask"

for dir in os.listdir(FROM):
    if dir == ".DS_Store" or dir == "image_dict.json":
        continue
    from_dir = os.path.join(FROM, dir)
    to_dir = os.path.join(TO, dir)
    for npy in os.listdir(from_dir):
        if npy == ".DS_Store":
            continue
        if "_0_0_0_0_0_" in npy or "_0_0_0_brightness_0_" in npy:
            shutil.copy(os.path.join(from_dir, npy), os.path.join(to_dir, npy))
