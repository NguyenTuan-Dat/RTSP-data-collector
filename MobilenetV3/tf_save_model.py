import numpy as np
import tensorflow as tf
import keras2onnx
import onnxruntime
from mobilenetv3 import MobileNetV3

model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax"
)
model.load_weights("/Users/ntdat/Downloads/MobileNetV3_large_500 (1).h5")

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)
