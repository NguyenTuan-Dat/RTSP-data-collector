from openvino.inference_engine import IENetwork
import numpy as np
from cv2 import resize


class HumanPose(object):
    def __init__(self, ie, xml_path, bin_path, device='CPU'):
        self.model = IENetwork(xml_path, bin_path)

        self._input_blob = next(iter(self.model.inputs))
        self._output_blob = next(iter(self.model.outputs))

        self._ie = ie
        self._exec_model = self._ie.load_network(network=self.model, device_name=device)

    def _infer(self, image):
        output = self._exec_model.infer(inputs={self._input_blob: image})
        return output

    def _preprocess(self, image):
        processed_image = np.expand_dims(resize(image, (448, 256)), axis=0)
        processed_image = processed_image.transpose((0, 3, 1, 2))
        return processed_image

    def detect(self, image):
        image = self._preprocess(image)
        infer_output = self._infer(image)

        return infer_output
