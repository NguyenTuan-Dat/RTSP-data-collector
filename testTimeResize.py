import cv2
import time

img = cv2.imread("/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Dự án công ty/Test/data_collection/6/6_0_0_0_0_0_1622001769.882365.jpg")
t = time.time()
img_resize = cv2.resize(img, (300,300), interpolation=cv2.INTER_LINEAR)
print("cv2.INTER_LINEAR: {}".format(time.time()-t))
t = time.time()
img_resize = cv2.resize(img, (300,300), interpolation=cv2.INTER_AREA)
print("cv2.INTER_AREA: {}".format(time.time()-t))
t = time.time()
img_resize = cv2.resize(img, (300,300), interpolation=cv2.INTER_CUBIC)
print("cv2.INTER_CUBIC: {}".format(time.time()-t))
