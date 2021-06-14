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
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(path_to_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.compat.v1.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def read_files(path_to_dir):
    images = []
    labels = []
    for dir in os.listdir(path_to_dir):
        path_to_folder = os.path.join(path_to_dir, dir)
        if ".DS_Store" == dir:
            continue
        for img_name in os.listdir(path_to_folder):
            if img_name == ".DS_Store":
                continue
            label = -1
            if "Glass" in dir:
                label = 1
            if "Normal" in dir:
                label = 0

            img = cv2.imread(os.path.join(path_to_folder, img_name))
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)

    images, labels = np.array(images), np.array(labels)

    print(images.shape, labels.shape)
    return images, labels


PATH_TO_DIR = "/Users/ntdat/Downloads/faces-spring-2020"
images, labels = read_files(PATH_TO_DIR)
convert_to(PATH_TO_DIR, images[:4500], labels[:4500], "faces-spring-2020-train")
convert_to(PATH_TO_DIR, images[4500:], labels[4500:], "faces-spring-2020-test")
