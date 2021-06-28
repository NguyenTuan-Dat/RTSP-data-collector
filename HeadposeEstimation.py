"""
Must activate openvino enviroments first:
SETUP_OPENVINO_PATH = '/opt/intel/openvino_2021.2.185/bin/setupvars.sh'
command = 'source /opt/intel/openvino_2021.2.185/bin/setupvars.sh'
"""
from numpy.core.numeric import count_nonzero
from openvino.inference_engine import IECore, IENetwork

import numpy as np
from cv2 import resize

class HeadposeEstimation(object):
    def __init__(self, ie, xml_path, bin_path, input_shape = (60, 60), device = 'CPU', thresold= 0.5):

        self.model = IENetwork(xml_path, bin_path)
        self._thresold = thresold

        self._input_blob = next(iter(self.model.inputs))

        self._input_shape = input_shape
        self._ie = ie
        self._exec_model = self._ie.load_network(network=self.model, device_name= device)

    def _infer(self, image):
        output = self._exec_model.infer(inputs={self._input_blob: image})
        return output

    def _preprocess(self, image):
        processed_image = np.expand_dims(resize(image, self._input_shape), axis=0)
        processed_image = processed_image.transpose((0, 3, 1, 2))
        return processed_image

    def _postprocess(self, infer_output):
        yaw = infer_output['angle_p_fc'][0][0]
        pitch = infer_output['angle_p_fc'][0][0]
        roll = infer_output['angle_r_fc'][0][0]
        return yaw, pitch, roll
    
    def detect(self, image):
        image = self._preprocess(image)
        infer_output = self._infer(image)

        yaw, pitch, roll = self._postprocess(infer_output)
        return yaw, pitch, roll

    
    