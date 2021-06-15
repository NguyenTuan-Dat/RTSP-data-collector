import os
import tensorflow as tf
from Unet import Unet

BATCH_SIZE = 8
IMG_SIZE = 224
NUM_CLASSES = 2
NUM_EPOCH = 100


@tf.function
def train_parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}

    features = tf.io.parse_single_example(example_proto, features)

    print(features['image_raw'])

    # You can do more image distortion here for training data
    img = tf.image.decode_png(features['image_raw'], channels=3)
    img = tf.reshape(img, (224, 224, 3))
    h, w, c = img.shape
    if h != 224 or w != 224 or c != 3:
        assert 0, "Assert! Input image shape should be (224, 224, 3)!!!"
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def loss_cce(label, predict):
    return tf.reduce_mean(
        tf.keras.backend.categorical_crossentropy(target=tf.cast(label, dtype=tf.float32), output=predict))


input_tensor = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
label_tensor = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))
predict_tensor = Unet(input_tensor)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

cce = loss_cce(label=label_tensor, predict=predict_tensor)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for epoch in range(NUM_EPOCH):
    for i in range(0, len(list_label) + 1, BATCH_SIZE):
        batch_labels = []
        batch_images = []
        if i + BATCH_SIZE >= len(list_label):
            break
        for j in range(BATCH_SIZE):
