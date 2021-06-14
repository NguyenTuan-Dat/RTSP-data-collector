import tensorflow as tf
import os


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


train_tfrecords_f = os.path.join("/Users/ntdat/Downloads/faces-spring-2020-224x224/faces-spring-2020-train.tfrecords")
train_dataset = tf.data.TFRecordDataset(train_tfrecords_f)
(img, label) = train_dataset.map(train_parse_function)
print(train_dataset)
