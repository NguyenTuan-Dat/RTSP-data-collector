"""
Must activate openvino enviroments first:
SETUP_OPENVINO_PATH = '/opt/intel/openvino_2021.2.185/bin/setupvars.sh'
command = 'source /opt/intel/openvino_2021.2.185/bin/setupvars.sh'
"""
from numpy.core.numeric import count_nonzero
from openvino.inference_engine import IECore, IENetwork

import numpy as np
from cv2 import resize


class FaceDetection(object):
    def __init__(self, ie, xml_path, bin_path, device='CPU'):

        self.model = IENetwork(xml_path, bin_path)

        self._input_blob = next(iter(self.model.inputs))
        self._output_blob = next(iter(self.model.outputs))

        self._ie = ie
        self._exec_model = self._ie.load_network(network=self.model, device_name=device)

    def _infer(self, image):
        output = self._exec_model.infer(inputs={self._input_blob: image})
        return output['detection_out']

    def _preprocess(self, image):
        processed_image = np.expand_dims(resize(image, (300, 300)), axis=0)
        processed_image = processed_image.transpose((0, 3, 1, 2))
        return processed_image

    def _postprocess(self, bboxes):
        new_bboxes = []
        for bbox in bboxes[0][0]:
            if bbox[2] > 0.9:
                new_bboxes.append(bbox[3:7])
        return new_bboxes

    def detect(self, image):
        image = self._preprocess(image)
        infer_output = self._infer(image)
        bboxes = self._postprocess(infer_output)
        return bboxes
