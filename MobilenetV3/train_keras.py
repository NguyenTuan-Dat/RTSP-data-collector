from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import os
import cv2

INPUT_SIZE = 224


def custom_mean_squared_error(y_trues, y_pred):
    y_t = []
    for y_true in y_trues:
        if y_true == 1:
            y_t.append([0, 1])
        else:
            y_t.append([1, 0])
    y_true = np.asarray(y_t)
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_phase', type=bool, default=True, help='train phase, true or false!')
    parser.add_argument('--model_type', type=str, default="large", help='model type, choice large or small!')
    parser.add_argument('--max_epoch', default=30, help='max epoch to train the network！')
    parser.add_argument('--input_shape', default=(224, 224, 3), help='the input size！')
    parser.add_argument('--classes_number', type=int, default=2, help='class number depend on your training datasets！')
    parser.add_argument('--weight_decay', default=2e-4, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.',
                        default=[1e-4, 1e-5, 5e-6, 1e-6])
    parser.add_argument('--train_batch_size', default=32, help='batch size of training.')
    parser.add_argument('--test_batch_size', default=32, help='batch size of testing.')
    parser.add_argument('--train_tfrecords_file_path', default='/content/Data/faces-spring-2020-train.tfrecords',
                        type=str,
                        help='path to the training datasets of tfrecords file path')
    parser.add_argument('--test_tfrecords_file_path', default='/content/Data/faces-spring-2020-test.tfrecords',
                        type=str,
                        help='path to the testing datasets of tfrecords file path')
    parser.add_argument('--ckpt_path', default='/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/',
                        help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/',
                        help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./logs', help='the log file save path')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--ckpt_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--dropout_rate', type=float, help='dropout rate', default=0.2)

    args = parser.parse_args()

    return args


def evaluation(log_dir, datasets, model, summary_writer, loss_fn, lr, step):
    count = 0
    loss_count = 0
    total_predict = []
    total_loss = 0

    def batch_evaluation(pred, labels):
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        return list(correct_prediction.numpy())

    for i, (images, labels) in enumerate(datasets):
        logits = model(images, training=False)
        pred = tf.nn.softmax(logits)
        loss_value = loss_fn(labels, pred)
        total_loss += loss_value
        loss_count += 1
        batch_correct_prediction = batch_evaluation(pred, labels)
        total_predict.extend(batch_correct_prediction)
        count += len(labels)

    img_1 = cv2.imread("/content/face-3179.png")
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = cv2.resize(img_1, (INPUT_SIZE, INPUT_SIZE))

    img_2 = cv2.imread("/content/face-860.png")
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_2 = cv2.resize(img_2, (INPUT_SIZE, INPUT_SIZE))

    img = []
    img.append(img_1)
    img.append(img_2)

    img = np.asarray(img)

    print(img.shape)

    logits = model(img, training=False)
    pred = tf.nn.softmax(logits)
    batch_correct_prediction = batch_evaluation(pred, [1, 0])
    print("pred_test", pred)
    print("logits_test", logits)
    print("batch_correct_prediction_test", batch_correct_prediction)

    total_predict = np.asarray(total_predict)
    print(total_predict)
    Accuracy = tf.reduce_mean(total_predict)
    mean_loss = total_loss / loss_count
    print(f'test total images {count}, Accuracy is {Accuracy}, Mean loss is {mean_loss}, lr is {lr}!')

    with open(os.path.join(log_dir, 'result.txt'), 'at') as f:
        f.write(f'test total images {count}, Accuracy is {Accuracy}, Mean loss is {mean_loss}, lr is {lr}!\n')

    with summary_writer.as_default():
        tf.summary.scalar('train/eval_loss', mean_loss, step=step)
        tf.summary.scalar('train/eval_accuracy', Accuracy, step=step)


@tf.function
def train_parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_png(features['image_raw'])
    img = tf.reshape(img, (INPUT_SIZE, INPUT_SIZE, 3))
    h, w, c = img.shape
    if h != INPUT_SIZE or w != INPUT_SIZE or c != 3:
        assert 0, "Assert! Input image shape should be (INPUT_SIZE, INPUT_SIZE, 3)!!!"
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


@tf.function
def test_parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, (INPUT_SIZE, INPUT_SIZE, 3))
    h, w, c = img.shape
    if h != INPUT_SIZE or w != INPUT_SIZE or c != 3:
        assert 0, "Assert! Input image shape should be (INPUT_SIZE, INPUT_SIZE, 3)!!!"
    label = tf.cast(features['label'], tf.int64)
    return img, label


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()

    lr_schedule = [1e-3, 1e-4, 5e-5, 1e-5, 1e-6]

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    # create log dir
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join("output", subdir, os.path.expanduser(args.log_file_path))

    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    output_dir = os.path.join("output", subdir, os.path.expanduser(args.ckpt_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_writer = tf.summary.create_file_writer(log_dir)  # create summary file writer

    # training datasets pipe
    train_tfrecords_f = os.path.join(args.train_tfrecords_file_path)
    train_dataset = tf.data.TFRecordDataset(train_tfrecords_f)
    train_dataset = train_dataset.map(train_parse_function)
    # dataset = dataset.shuffle(buffer_size=args.buffer_size)
    train_dataset = train_dataset.batch(args.train_batch_size)
    # testing datasets pipe
    test_tfrecords_f = os.path.join(args.test_tfrecords_file_path)
    test_dataset = tf.data.TFRecordDataset(test_tfrecords_f)
    test_dataset = test_dataset.map(test_parse_function)
    test_dataset = test_dataset.batch(args.test_batch_size)

    epoch_var = tf.Variable(0, trainable=False)
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=args.lr_schedule,
                                                                            values=lr_schedule,
                                                                            name='lr_schedule')

    lr = learning_rate_fn(epoch_var)

    # Instantiate an optimizer, special change optimizers in here if need.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # Instantiate a loss function， customer loss function can insert in here.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax"
    )

    model.compile(optimizer=optimizer, loss=custom_mean_squared_error,
                  metrics=[tf.compat.v1.keras.metrics.SparseCategoricalAccuracy()])

    tf.keras.backend.set_learning_phase(True)
    # model architecture write to file
    fd = open(f'./misc/MobileNetV3_{args.model_type}.txt', "w")
    for var in model.variables:
        info = f'<tf.Variable \'{var.name}\'' + f' shape={var.shape}' + f' dtype={var.numpy().dtype}>'
        fd.write(info + "\n")
    fd.close()

    # load model weights
    if args.pretrained_model:
        model.load(args.pretrained_model)
        # model.load_weights(args.pretrained_model)
        print(f'Successful to load pretrained model!')
    step = 0

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f'MobileNetV3_{args.model_type}_{step}.h5'), save_freq=100
        )
    ]

    model.fit(train_dataset=train_dataset, epochs=10, callbacks=callbacks, validation_data=test_dataset)

    ckpt_path = os.path.join(output_dir, f'MobileNetV3_final.h5')
    tf.saved_model.save(model, os.path.join(output_dir, f'MobileNetV3_final'))
    # final test
    evaluation(log_dir, test_dataset, model, summary_writer, loss_fn, lr, step)
