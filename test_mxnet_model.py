import cv2
from TestTrain.fmobilenetv3 import MobileNetV3
import numpy as np
import time
from openvino.inference_engine import IECore
from face_detection import FaceDetection
import mxnet as mx

INPUT_SIZE = 224

# path to models
FACEDETECTION_XML_PATH = "./models/face-detection-retail-0004.xml"
FACEDETECTION_BIN_PATH = "./models/face-detection-retail-0004.bin"

ie = IECore()

# Create FaceDetection model
facedetection = FaceDetection(ie, FACEDETECTION_XML_PATH, FACEDETECTION_BIN_PATH)

# model = tf.keras.applications.MobileNetV3Small(
#     input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
#     alpha=1.0,
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     pooling=None,
#     classes=2,
#     classifier_activation="softmax"
# )

model = MobileNetV3(num_classes=2, mode="small")
model.load_parameters("/Users/ntdat/Downloads/net_9 (1).params", ctx=mx.cpu())


# img = cv2.imread("/Users/ntdat/Downloads/faces-spring-2020/Normal/face-2690.png")

def process_image(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0)
    img = img.copyto(mx.cpu(0)).as_nd_ndarray()

    logits = model(img)
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

                result = np.asarray(result)

                print(result)

                if result[0][0] > 0.6:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.putText(frame, text=str(result[0][0]) + "size:" + str(y_max - y_min), org=(x_min, y_min),
                            fontFace=cv2.INTER_AREA, fontScale=1,
                            color=color)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color)

            except Exception as ex:
                print(ex)

            print("FPS: {}".format(1 / (time.time() - t)), end="\r")

    cv2.imshow("AloAlo", frame)
    cv2.waitKey(1)

# for layer in model.layers:
#     print(layer.get_weights())
