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

NUM_CLASSES = 2


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


mnist_train = mx.gluon.data.vision.datasets.ImageFolderDataset(
    '/content/RTSP-data-collector/TestTrain/faces-spring-2020-224_mxnet/train')

transformer = transforms.Compose([transforms.ToTensor()])

batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=4)

for data, label in train_data:
    print(data.shape, label.shape)
    break

mnist_valid = mx.gluon.data.vision.datasets.ImageFolderDataset(
    '/content/RTSP-data-collector/TestTrain/faces-spring-2020-224_mxnet/test')
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

net = MobileNetV3(input_shape=(3, 28, 28), classes=NUM_CLASSES, mode="small")
net.initialize(init=init.Xavier(), ctx=npx.gpu(0))
net.hybridize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

print(train_data)
print(trainer)

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        data *= 255
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
        data *= 255
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data),
        valid_acc / len(valid_data), time.time() - tic))
    net.export("/content/drive/MyDrive/Colab Notebooks/net_adam_{}".format(epoch), epoch=1)

# sym = get_symbol(input_shape=(3, 224, 224), num_classes=NUM_CLASSES, mode="small")
#
# image = cv2.imread("/Users/ntdat/Downloads/faces-spring-2020-224x224/Glass/2img_44.jpg")
# image = cv2.resize(image, (224, 224))
# image = np.transpose(image, (2, 0, 1))
#
# model = mx.mod.Module(
#     context=[mx.cpu()],
#     symbol=sym,
#     label_names=['data']
# )
#
# model.bind(for_training=False, data_shapes=data_iter.provide_data)
#
# print(model.forward(image))
