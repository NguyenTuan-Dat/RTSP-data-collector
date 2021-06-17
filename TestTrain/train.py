from fmobilenetv3 import get_symbol
from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, autograd, init, np, npx
from IPython import display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import mxnet as mx
from fmobilenetv3 import MobileNetV3
import time
import os
import argparse
import mobilenetv3
from base_unet import BaseUnet

parser = argparse.ArgumentParser()
parser.add_argument("-nn", type=str, default='mobilenetv3', choices=['mobilenetv3', 'fmobilenetv3', 'base_unet'])
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--num_epoch", type=int, default=10)
args = parser.parse_args()

NUM_CLASSES = args.num_classes
INPUT_SHAPE = args.input_shape


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


path_to_data_resized = '/content/faces-spring-2020-224_mxnet'

mnist_train = mx.gluon.data.vision.datasets.ImageFolderDataset(
    os.path.join(path_to_data_resized, 'train'))

transformer = transforms.Compose([transforms.Resize(INPUT_SHAPE), transforms.ToTensor()])

batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=4)

for data, label in train_data:
    print(data.shape, label.shape)
    break

mnist_valid = mx.gluon.data.vision.datasets.ImageFolderDataset(
    os.path.join(path_to_data_resized, 'test'))
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)
net = None

# get neural network
if args.nn == 'fmobilenetv3':
    net = MobileNetV3(classes=NUM_CLASSES, mode="small")
elif args.nn == 'mobilenetv3':
    net = mobilenetv3.MobileNetV3(version='small', num_classes=NUM_CLASSES)
elif args.nn == 'base_unet':
    net = BaseUnet(num_classes=NUM_CLASSES)

net.initialize(init=init.Xavier(), ctx=npx.gpu(0))
net.hybridize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

for epoch in range(args.num_epoch):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in valid_data:
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data),
        valid_acc / len(valid_data), time.time() - tic))
    net.export("/content/drive/MyDrive/Colab Notebooks/{}_{}_{}".format(args.nn, INPUT_SHAPE, epoch), epoch=1)
