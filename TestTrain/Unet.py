import tensorflow as tf
import numpy as np

NUM_CLASSES = 2


def Unet(input_image):
    '''
    :param input_image: Input shape = (N, 224, 224, 3)
    :return:
    '''
    # block1
    conv1_1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), strides=1, padding="same")
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 224, 224, 64)

    # block2
    maxP2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, padding="same")
    conv2_1 = tf.layers.conv2d(inputs=maxP2, filters=128, kernel_size=(3, 3), strides=1, padding="same")
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 112, 112, 128)

    # block3
    maxP3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, padding="same")
    conv3_1 = tf.layers.conv2d(inputs=maxP3, filters=256, kernel_size=(3, 3), strides=1, padding="same")
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 56, 56, 256)

    # block4
    maxP4 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=(2, 2), strides=2, padding="same")
    conv4_1 = tf.layers.conv2d(inputs=maxP4, filters=512, kernel_size=(3, 3), strides=1, padding="same")
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 28, 28, 512)

    # block5
    maxP5 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=(2, 2), strides=2, padding="same")
    conv5_1 = tf.layers.conv2d(inputs=maxP5, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 14, 14, 1024)

    # block6
    maxP6 = tf.layers.max_pooling2d(inputs=conv5_2, pool_size=(2, 2), strides=2, padding="same")
    conv6_1 = tf.layers.conv2d(inputs=maxP6, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
    conv6_2 = tf.layers.conv2d(inputs=conv6_1, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 7, 7, 1024)

    # block7
    maxP7 = tf.layers.max_pooling2d(inputs=conv6_2, pool_size=(7, 7), strides=0, padding="same")
    # Shape = (N, 1, 1, 1024)

    output_classify = tf.layers.conv2d(inputs=maxP7, filters=NUM_CLASSES, kernel_size=(3, 3), strides=1,
                                       padding="same")

    return tf.nn.softmax(output_classify)
