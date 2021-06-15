import numpy
import tensorflow as tf
import os
import cv2
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# images and labels array as input
def convert_to(path_to_dir, images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))

    filename = os.path.join(path_to_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.compat.v1.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(int(labels[index]))}))
        writer.write(example.SerializeToString())


def read_files(path_to_dir):
    images = []
    labels = []
    for dir in os.listdir(path_to_dir):
        path_to_folder = os.path.join(path_to_dir, dir)
        if ".DS_Store" == dir:
            continue
        for img_name in os.listdir(path_to_folder):
            if ".DS_Store" == img_name:
                continue
            label = -1
            if "Glass" in dir:
                label = 1
            if "Normal" in dir:
                label = 0
            with open(os.path.join(path_to_folder, img_name), 'rb') as f:
                img = f.read()
            images.append(img)
            labels.append(label)

    images, labels = np.array(images), np.array(labels)

    concat = np.stack((images, labels), axis=1)

    np.random.shuffle(concat)

    concat = concat.T

    print(concat.shape)

    images, labels = concat[0], concat[1]

    # print(images, labels)

    # for i, label in enumerate(labels):
    #     nparr = np.fromstring(images[i], np.uint8)
    #     img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     cv2.putText(img_np, str(label), (50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0))
    #     cv2.imshow("AloAlo", img_np)
    #     cv2.waitKey()

    print(images.shape, labels.shape)

    return images, labels


PATH_TO_DIR = "/Users/ntdat/Downloads/faces-spring-2020-224x224"
images, labels = read_files(PATH_TO_DIR)
convert_to(PATH_TO_DIR, images[:7000], labels[:7000], "faces-spring-2020-train")
convert_to(PATH_TO_DIR, images[7000:], labels[7000:], "faces-spring-2020-test")
