"""
Must activate openvino enviroments first:
SETUP_OPENVINO_PATH = '/opt/intel/openvino_2021.2.185/bin/setupvars.sh'
command = 'source /opt/intel/openvino_2021.2.185/bin/setupvars.sh'
"""
from numpy.core.numeric import count_nonzero
from openvino.inference_engine import IECore, IENetwork

import numpy as np
from cv2 import resize


class GlassMobilenet(object):
    def __init__(self, ie, xml_path, bin_path, device='CPU', input_shape=224):
        self.model = IENetwork(xml_path, bin_path)

        self._input_blob = next(iter(self.model.inputs))
        self._output_blob = next(iter(self.model.outputs))

        self.input_shape = input_shape

        self._ie = ie
        self._exec_model = self._ie.load_network(network=self.model, device_name=device)

    def _infer(self, image):
        output = self._exec_model.infer(inputs={self._input_blob: image})
        return output

    def _preprocess(self, image):
        try:
            processed_image = np.expand_dims(resize(image, (self.input_shape, self.input_shape)), axis=0)
            processed_image = processed_image.transpose((0, 3, 1, 2))
            return processed_image
        except Exception as ex:
            print("preprocess", ex)
        return False

    def detect(self, image):
        image = self._preprocess(image)
        infer_output = self._infer(image)
        return infer_output
