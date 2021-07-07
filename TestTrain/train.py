from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

from CustomMobilenet import MobileNetV2
from fmobilenetv3 import get_symbol
from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, autograd, init, np, npx, is_np_array
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
import gluoncv

parser = argparse.ArgumentParser()
parser.add_argument("-nn", type=str, default='mobilenetv3',
                    choices=['mobilenetv3', 'fmobilenetv3', 'gmobilenetv3', 'base_unet', 'mobilenetv2_50',
                             "mobilenetv2_25", 'alexnet'])
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--num_epoch", type=int, default=10)
args = parser.parse_args()

NUM_CLASSES = args.num_classes
INPUT_SHAPE = args.input_shape

if not os.path.exists("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format(args.nn)):
    os.mkdir("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format(args.nn))


class CustomLoss(Loss):
    def __init__(self, num_classes=3, focus_classes=(0, 1, 2), weight=None, batch_axis=0, **kwargs):
        super(CustomLoss, self).__init__(weight, batch_axis, **kwargs)
        self.num_classes = num_classes
        self.focus_classes = np.array(focus_classes)

    def hybrid_forward(self, F, preds, labels):
        losses = None
        for idx, pred in enumerate(preds):
            loss = F.abs(labels[idx] - pred)
            # loss = labels[idx] * F.log(pred + 1e-10)
            loss = loss.sum()
            loss = loss if mx.nd.argmax(labels[idx]) != 0 else loss * 2
            if idx == 0:
                losses = loss
            else:
                losses = mx.nd.concat(losses, loss, dim=0)

        # print("pred," , preds[0], "label,", labels[0], "losses,",losses[0] )
        return losses


class Softmax(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        softmax = F.npx.softmax if is_np_array() else F.softmax
        return softmax(x)

    def __repr__(self):
        return self.__class__.__name__


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype("float32")).mean().asscalar()


path_to_data_resized = '/content/faces-spring-2020-224_mxnet'

mnist_train = mx.gluon.data.vision.datasets.ImageFolderDataset(
    os.path.join(path_to_data_resized, 'train'))

transformer = transforms.Compose([transforms.Resize(INPUT_SHAPE),
                                  transforms.RandomBrightness(0.5),
                                  transforms.RandomFlipLeftRight(),
                                  transforms.RandomSaturation(0.5),
                                  transforms.ToTensor()
                                  ])

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
elif args.nn == 'gmobilenetv3':
    kwargs = {'ctx': mx.gpu(), 'pretrained': True, 'classes': 3, 'last_gamma': True}
    net = gluoncv.model_zoo.get_model("mobilenetv3_small", classes=NUM_CLASSES)
elif args.nn == 'base_unet':
    net = BaseUnet(num_classes=NUM_CLASSES)
elif args.nn == 'mobilenetv2_50':
    net = gluon.model_zoo.vision.MobileNetV2(classes=NUM_CLASSES, multiplier=0.5)
    net.output.add(Softmax())
elif args.nn == 'mobilenetv2_25':
    net = gluon.model_zoo.vision.MobileNetV2(classes=NUM_CLASSES, multiplier=0.25)
    net.output.add(Softmax())
elif args.nn == 'alexnet':
    net = gluon.model_zoo.vision.AlexNet(classes=NUM_CLASSES)

net.initialize(init=init.Xavier(), ctx=npx.gpu(0))
net.hybridize()
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# sigmoid_cross_entropy = mx.gluon.loss.L2Loss()
custom_loss = CustomLoss(focus_classes=(0,))
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0005})

for epoch in range(args.num_epoch):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        label_onehot = mx.nd.one_hot(label, 3)
        label_onehot = label_onehot.copyto(mx.gpu(0)).as_nd_ndarray()
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        with autograd.record():
            output = net(data)
            loss_softmax_ce = softmax_cross_entropy(output, label)

            loss = loss_softmax_ce

            loss_custom = custom_loss(output, label_onehot)
            loss = 0.7 * loss_softmax_ce.mean() + 0.3 * loss_custom.mean()
        loss.backward()

        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in valid_data:
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        # label = mx.nd.one_hot(label, 3)
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data),
        valid_acc / len(valid_data), time.time() - tic))
    net.export(
        "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}/{}_{}_{}".format(args.nn, args.nn,
                                                                                                 INPUT_SHAPE, epoch),
        epoch=1)
