import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import time
from openvino.inference_engine import IECore
from face_detection import FaceDetection

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)


def batch_evaluation(pred, labels):
    correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
    return list(correct_prediction.numpy())


model = model_from_json(open("/Users/ntdat/Downloads/" + 'model.json').read())
model.load_weights("/Users/ntdat/Downloads/" + 'MobileNetV3_large_1500.h5')
img = cv2.imread("/Users/ntdat/Downloads/faces-spring-2020/Glass/face-2319.png")


# img = cv2.imread("/Users/ntdat/Downloads/faces-spring-2020/Normal/face-2690.png")

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    logits = model(img, training=False)
    return logits


video = cv2.VideoCapture(0)

while (video.isOpened()):
    _, frame = video.read()

    h, w, c = frame.shape

    # get face on frame
    outputs = facedetection.detect(frame)

    if len(outputs) != 0:
        outputs = np.array(outputs)
        color = (0, 255, 0)
        for output in outputs:
            t = time.time()
            try:
                # get face location
                x_min, y_min, x_max, y_max = (output * [w, h, w, h]).astype(int)

                if x_max - x_min > y_max - y_min:
                    y_max += ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min -= ((x_max - x_min) - (y_max - y_min)) / 2
                    y_min = 0 if y_min < 0 else int(y_min)
                    y_max = h if y_max > h else int(y_max)
                else:
                    x_max += ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min -= ((y_max - y_min) - (x_max - x_min)) / 2
                    x_min = 0 if x_min < 0 else int(x_min)
                    x_max = w if x_max > w else int(x_max)

                # crop face
                img_cropped = frame[y_min:y_max, x_min:x_max]

                # img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

                result = process_image(img_cropped)

                print(result)

                cv2.putText(frame, text=str(result), org=(x_min, y_min), fontFace=cv2.INTER_AREA, fontScale=1,
                            color=color)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)

            except Exception as ex:
                print(ex)

            print("FPS: {}".format(1 / (time.time() - t)), end="\r")

    cv2.imshow("AloAlo", frame)
    cv2.waitKey(1)

# for layer in model.layers:
#     print(layer.get_weights())
