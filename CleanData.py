import cv2
import os

PATH_TO_FOLDER = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Dự án công ty/Test/data_collection/6/"

img_names = os.listdir(PATH_TO_FOLDER)
img_name = img_names.sort()

list_del = []
idx = 0

while idx < len(img_names):

    img_name = img_names[idx]
    img = cv2.imread(PATH_TO_FOLDER + img_name)
    print("{:4}, Showing: {}".format(idx, img_name), end="\r")
    try:
        cv2.imshow("AloAlo", img)
    except Exception as ex:
        print(ex)
    key = cv2.waitKey()
    if key == ord("q"):
        break
    if key == ord("."):
        if idx < len(img_names):
            idx += 1
    if key == ord(","):
        idx -= 1
    if key == ord('d') or key == ord('D'):
        if img_name not in list_del:
            print("Removing: ", img_name)
            list_del.append(img_name)
        else:
            print("List del has {}".format(img_name))
    if key == ord('r') or key == ord('R'):
        if img_name in list_del:
            print("Revert from delete quere: ", img_name)
            list_del.remove(img_name)
        else:
            print("List del don't have {}".format(img_name))

for del_name in list_del:
    print("Removing: ", del_name)
    os.remove(PATH_TO_FOLDER + del_name)

print("Deleted: {} image in folder: {}".format(len(list_del), PATH_TO_FOLDER))
cv2.destroyAllWindows()