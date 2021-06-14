import os

PATH_TO_FOLDER = ""

list_img = os.listdir(PATH_TO_FOLDER)

img_name = "24_0_0_mask_0_2_1621907032.5584013"
time_to_rename = 1621907032.5584013
for img_name in list_img:
    if "24_0_0_mask_0_" in img_name:
        img_time = float(img_name.split("_")[-1].split(".jpg")[0])
        if img_time >= time_to_rename:
            print("24_hat_0_0_0_" + img_name.split("_mask_0_")[-1])
            os.rename(PATH_TO_FOLDER + img_name, PATH_TO_FOLDER + "24_hat_0_0_0_" + img_name.split("_mask_")[-1])