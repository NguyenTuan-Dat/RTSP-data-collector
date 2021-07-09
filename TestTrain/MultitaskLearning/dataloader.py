import mxnet as mx
import os
from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon


class MultitaskDataLoader:
    def __init__(self, batch_size, input_shape, path_to_data_folder):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.path_to_data_folder = path_to_data_folder
        self.transformer = transforms.Compose([transforms.Resize(self.input_shape),
                                               transforms.RandomBrightness(0.5),
                                               transforms.RandomFlipLeftRight(),
                                               transforms.RandomSaturation(0.5),
                                               transforms.ToTensor()
                                               ])

    def get_data(self):
        mnist = mx.gluon.data.vision.datasets.ImageFolderDataset(self.path_to_data_folder)
        data = gluon.data.DataLoader(
            mnist.transform_first(self.transformer), batch_size=self.batch_size, shuffle=True, num_workers=4)

        for input, label in data:
            multitask_labels = []
            for idx, i in enumerate(input):
                multitask_label = mx.ndarray.zeros((3, 2))
                multitask_labels[i][1] = 1
                multitask_labels.append(multitask_labels)

            multitask_labels = mx.nd.array(multitask_labels)
            datasets = mx.gluon.data.datasets.ArrayDataset(input, multitask_labels)

        return datasets
