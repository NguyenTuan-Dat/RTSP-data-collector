import cv2
import os

accuracy_logs_content = ""
with open("/Users/ntdat/Downloads/logs_HN.txt", "r") as f:
    accuracy_logs_content = f.read()

for idx, line in enumerate(accuracy_logs_content.split("\n")):
    try:
        file_name, distance = line.split(" Predict user_id: ")
        pred_id = distance.split(" ")[0]
        file_name = file_name.split(": ")[-1]
        distance = float(distance.split(": ")[-1].split(" is ")[0])

        print("/Users/ntdat/Downloads/HN/cropped/" + file_name + ".jpg",
              "/Users/ntdat/Downloads/cropped/" + pred_id + ".jpg")

        hn = cv2.imread("/Users/ntdat/Downloads/HN/cropped/" + file_name + ".jpg")
        dn = cv2.imread("/Users/ntdat/Downloads/cropped/" + pred_id + ".jpg")

        print(hn.shape, dn.shape)

        if dn == None:
            print("dnnnn", pred_id)

        if hn == None:
            print("hnnnn")

        dn = cv2.resize(dn, (400, 400))
        hn = cv2.resize(hn, (400, 400))

        img = cv2.hconcat([hn, dn])
        cv2.putText(img, "{:.3f}".format(distance), (50, 50), fontFace=cv2.FONT_ITALIC, fontScale=2, color=(0, 255, 0))
        cv2.imwrite("/Users/ntdat/Downloads/HN/" + file_name + ".jpg", img)
    except Exception as ex:
        print(ex)
