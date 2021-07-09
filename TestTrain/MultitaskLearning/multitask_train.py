from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, autograd, init, np, npx, is_np_array
import time
import os
import argparse
from dataloader import MultitaskDataLoader
import mxnet as mx

parser = argparse.ArgumentParser()
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--num_epoch", type=int, default=10)
args = parser.parse_args()

NUM_CLASSES = args.num_classes
INPUT_SHAPE = args.input_shape

if not os.path.exists(
        "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format("mobilenetv2_50")):
    os.mkdir("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format("mobilenetv2_50"))

path_to_data_resized = '/content/faces-spring-2020-224_mxnet'

mnist_train = mx.gluon.data.vision.datasets.ImageFolderDataset(
    os.path.join(path_to_data_resized, 'train'))

print(mnist_train)

transformer = transforms.Compose([transforms.Resize(INPUT_SHAPE),
                                  transforms.RandomBrightness(0.5),
                                  transforms.RandomFlipLeftRight(),
                                  transforms.RandomSaturation(0.5),
                                  transforms.ToTensor()
                                  ])

batch_size = 256
train_data = MultitaskDataLoader(batch_size=batch_size,
                                 input_shape=INPUT_SHAPE,
                                 path_to_data_folder=os.path.join(path_to_data_resized, "train"))
train_data.get_data()

# mnist_valid = mx.gluon.data.vision.datasets.ImageFolderDataset(
#     os.path.join(path_to_data_resized, 'test'))
# valid_data = gluon.data.DataLoader(
#     mnist_valid.transform_first(transformer),
#     batch_size=batch_size, num_workers=4)
# net = None
#
# # get neural network
# net = gluon.model_zoo.vision.MobileNetV2(classes=NUM_CLASSES, multiplier=0.5)
# net.output.add(Softmax())
#
# net.initialize(init=init.Xavier(), ctx=npx.gpu(0))
# net.hybridize()
# softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# # sigmoid_cross_entropy = mx.gluon.loss.L2Loss()
# custom_loss = CustomLoss(focus_classes=(0,))
# trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0005})
#
# for epoch in range(args.num_epoch):
#     train_loss, train_acc, valid_acc = 0., 0., 0.
#     tic = time.time()
#     for data, label in train_data:
#         # forward + backward
#         data = data.copyto(mx.gpu(0)).as_nd_ndarray()
#         label_onehot = mx.nd.one_hot(label, 3)
#         label_onehot = label_onehot.copyto(mx.gpu(0)).as_nd_ndarray()
#         label = label.copyto(mx.gpu(0)).as_nd_ndarray()
#         with autograd.record():
#             output = net(data)
#             loss_softmax_ce = softmax_cross_entropy(output, label)
#
#             loss = loss_softmax_ce
#
#             loss_custom = custom_loss(output, label_onehot)
#             loss = 0.7 * loss_softmax_ce.mean() + 0.3 * loss_custom.mean()
#         loss.backward()
#
#         # update parameters
#         trainer.step(batch_size)
#         # calculate training metrics
#         train_loss += loss.mean().asscalar()
#         train_acc += acc(output, label)
#     # calculate validation accuracy
#     for data, label in valid_data:
#         data = data.copyto(mx.gpu(0)).as_nd_ndarray()
#         # label = mx.nd.one_hot(label, 3)
#         label = label.copyto(mx.gpu(0)).as_nd_ndarray()
#         valid_acc += acc(net(data), label)
#     print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
#         epoch, train_loss / len(train_data), train_acc / len(train_data),
#         valid_acc / len(valid_data), time.time() - tic))
#     net.export(
#         "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}/{}_{}_{}".format(args.nn, args.nn,
#                                                                                                  INPUT_SHAPE, epoch),
#         epoch=1)
