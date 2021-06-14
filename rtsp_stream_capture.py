import cv2
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', type=str)
parser.add_argument('-fps', '--fps', type=int, default=30)

args = parser.parse_args()

DATA_DIR = "data_collection"

path = os.path.join(DATA_DIR, args.name)
path += ".mp4"

cam_addr = "rtsp://ftech:ad1235min@192.168.130.27/live0"
cam = cv2.VideoCapture(cam_addr)
video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'H264'), args.fps, (1920, 1080))

while(cam.isOpened()):
    _, frame = cam.read()
    cv2.imshow("aloalo", frame)
    video.write(frame)
    key = cv2.waitKey(1)
    if ord('q') == key:
        break

video.release()
cam.release()
cv2.destroyAllWindows()