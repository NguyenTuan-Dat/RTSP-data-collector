import os
import shutil

FROM = "/Users/ntdat/Downloads/DownloadFromYoutube/Glasses_From_video"
TO = "/Users/ntdat/Downloads/DownloadFromYoutube/Glasses+Mask"

for idx, img_name in enumerate(os.listdir(FROM)):
    if idx % 5 == 0:
        shutil.copy(os.path.join(FROM, img_name), os.path.join(TO, img_name))
