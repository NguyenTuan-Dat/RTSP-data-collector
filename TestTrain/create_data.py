import mxnet as mx
import cv2

train_dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(
    '/Users/ntdat/Downloads/faces-spring-2020-224_mxnet/train')
test_dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(
    '/Users/ntdat/Downloads/faces-spring-2020-224_mxnet/test')

sample_idx = 888
sample = train_dataset[sample_idx]
data = sample[0]
label = sample[1]

cv2.imshow("aloalo", data.asnumpy())
cv2.waitKey()
print("Data type: {}".format(data.dtype))
print("Label: {}".format(label))
print("Label description: {}".format(train_dataset.synsets[label]))
assert label == 1
