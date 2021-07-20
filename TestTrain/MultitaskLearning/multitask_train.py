from mxnet.gluon.data.vision import datasets, transforms
from mxnet import gluon, autograd, init, np, npx, is_np_array
import time
import os
import argparse

from multitask_model import MobileNetV2Multitask
from dataloader import MultitaskDataLoader
import mxnet as mx

parser = argparse.ArgumentParser()
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--num_epoch", type=int, default=10)
args = parser.parse_args()

NUM_CLASSES = args.num_classes
INPUT_SHAPE = args.input_shape


def label2vec(labels):
    label_vecs = []
    for label in labels:
        vec = [0, 0, 0]  # Normal
        if label == 0:  # Glasses
            vec[0] = 1
        elif label == 1:  # Glasses + Hat
            vec[0] = 1
            vec[2] = 1
        elif label == 2:  # Glasses + Mask
            vec[0] = 1
            vec[1] = 1
        elif label == 3:  # Hat
            vec[2] = 1
        elif label == 4:  # Mask
            vec[1] = 1
        elif label == 5:  # Glasses + Mask + Hat
            vec[0] = 1
            vec[1] = 1
            vec[2] = 1
        elif label == 6:  # Mask + Hat
            vec[1] = 1
            vec[2] = 1
        label_vecs.append(vec)
    label_vecs = np.array(label_vecs)
    return label_vecs


def one_hot_label(label):
    label_onehot = label2vec(label).copyto(mx.gpu(0)).as_nd_ndarray()
    new_label = mx.nd.one_hot(label_onehot, 2, dtype=np.int8)
    return new_label


def acc(output, label_one_hot):
    glasses_acc = cal_acc(output[0], label_one_hot[1][0])
    mask_acc = cal_acc(output[1], label_one_hot[1][1])
    normal_acc = cal_acc(output[2], label_one_hot[1][2])

    return np.array([glasses_acc, mask_acc, normal_acc])


def cal_acc(output, label):
    return (output.argmax(axis=1) ==
            label.astype("float32")).mean().asscalar()


if not os.path.exists(
        "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format("mobilenetv2_50")):
    os.mkdir("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}".format("mobilenetv2_50"))

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
net = MobileNetV2Multitask(classes=NUM_CLASSES, multiplier=0.5)

net.initialize(init=init.Xavier(), ctx=npx.gpu(0))
net.hybridize()
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# sigmoid_cross_entropy = mx.gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0005})

for epoch in range(args.num_epoch):
    train_loss, train_acc, valid_acc = 0., np.array([0., 0., 0.]), np.array([0., 0., 0.])
    train_data = gluon.data.DataLoader(
        mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=4)
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        label_one_hot = one_hot_label(label).T
        with autograd.record():
            pred_glasses, pred_mask, pred_normal = net(data)
            loss_pred_glasses = softmax_cross_entropy(pred_glasses, label_one_hot[1][0])
            loss_pred_mask = softmax_cross_entropy(pred_mask, label_one_hot[1][1])
            loss_pred_normal = softmax_cross_entropy(pred_normal, label_one_hot[1][2])

            loss = loss_pred_glasses.mean() + loss_pred_mask.mean() + loss_pred_normal.mean()
        loss.backward()

        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc((pred_glasses, pred_mask, pred_normal), label_one_hot)
    # calculate validation accuracy
    for data, label in valid_data:
        data = data.copyto(mx.gpu(0)).as_nd_ndarray()
        label = label.copyto(mx.gpu(0)).as_nd_ndarray()
        label_one_hot = one_hot_label(label).T
        valid_acc += acc(net(data), label_one_hot)
    train_accuracy = np.array(train_acc) / len(train_data)
    valid_accuracy = np.array(valid_acc) / len(valid_data)
    print("Epoch {}: loss {:.3f}, train acc {}, test acc {}, in {:.1f} sec".format(epoch,
                                                                                   train_loss / len(train_data),
                                                                                   str(train_accuracy),
                                                                                   str(valid_accuracy),
                                                                                   time.time() - tic))
    if not os.path.exists("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_Multitask"):
        os.mkdir("/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_Multitask")
    net.export(
        "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_{}/{}_{}_{}".format("Multitask",
                                                                                                 "Multitask",
                                                                                                 INPUT_SHAPE,
                                                                                                 epoch),
        epoch=1)
