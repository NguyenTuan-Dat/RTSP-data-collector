from fmobilenetv3 import get_symbol
from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, autograd, init
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


# get data
mnist_train = datasets.FashionMNIST(train=True)
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
X, y = mnist_train[0:10]

# plot images
# display.set_matplotlib_formats('svg')
# _, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
# for f, x, yi in zip(figs, X, y):
#     # 3D->2D by removing the last channel dim
#     f.imshow(x.reshape((28, 28)).asnumpy())
#     ax = f.axes
#     ax.set_title(text_labels[int(yi)])
#     ax.title.set_fontsize(14)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])
mnist_train = mnist_train.transform_first(transformer)

batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

for data, label in train_data:
    print(data.shape, label.shape)
    break

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

net = MobileNetV3(input_shape=(3, 28, 28), num_classes=NUM_CLASSES, mode="small")
net.initialize(init=init.Xavier(), ctx=[mx.gpu(0)])
net.hybridize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
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
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), train_acc / len(train_data),
        valid_acc / len(valid_data), time.time() - tic))
    net.export("lenet", epoch=1)

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
