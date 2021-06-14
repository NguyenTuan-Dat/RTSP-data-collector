import numpy as np
import tensorflow as tf
from mobilenetv3 import MobileNetV3

model = MobileNetV3(type="large", input_shape=(224, 224, 3), classes_number=2,
                    l2_reg=2e-4, dropout_rate=0.2, name="MobileNetV3")
model.load_weights("/Users/ntdat/Downloads/MobileNetV3_final.h5")

tf.keras.models.save_model(model, "~/Downloads/model.ckpt")
