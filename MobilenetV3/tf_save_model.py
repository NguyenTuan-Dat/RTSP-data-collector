import keras2onnx
import onnxruntime
from tensorflow.keras.models import model_from_json

INPUT_SIZE = 224

model = model_from_json(open("/Users/ntdat/Downloads/models/" + 'model_{}.json'.format(INPUT_SIZE)).read())
model.load_weights("/Users/ntdat/Downloads/models/MobileNetV3_large_6500_224.h5")

# for layer in model.layers:
#     print(layer.get_weights())

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = '/Users/ntdat/Downloads/glass_large_6500_224.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)
