import cv2

cam_addr = "rtsp://ftech:ad1235min@192.168.130.27/live0"
video = cv2.VideoCapture(cam_addr)

while(video.isOpened()):
    _, frame = video.read()
    cv2.imshow("aloalo", frame)
    cv2.waitKey(1)