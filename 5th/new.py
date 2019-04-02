import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

N_CLASSES = 6
IMG_W = 28
IMG_H = 28
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 1
learning_rate = 0.0001


def get_file(file_dir):
    image_list = []
    label_list = []
    for train_class in os.listdir(file_dir):
        for pic in os.listdir(file_dir + '/' + train_class):
            image_list.append(file_dir + '/' + train_class + '/' + pic)
            label_list.append(train_class)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    label_list = [int(i) for i in label_list]
    return image_list, label_list


def get_batch(image, label, image_W, image_H):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    return image, label


def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv1(images):
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1) + b_conv1, name='conv1')
    return h_conv1


def pool1(h_conv1):
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pooling_2x2(h_conv1, 'pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    return norm1


def conv2(norm1):
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2, name='conv2')
    return h_conv2


def pool2(h_conv2):
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pooling_2x2(h_conv2, 'pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    return norm2


def conv3(norm2):
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3) + b_conv3, name='conv3')
    return h_conv3


def pool3(h_conv3):
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pooling_2x2(h_conv3, 'pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    return norm3


def fc1(norm3, batch_size):
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005), name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)
    return h_fc1


def fc2(h_fc1):
    with tf.variable_scope('local3') as scope:
        w_fc2 = tf.Variable(weight_variable([128, 128], 0.005), name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2, name=scope.name)
    return h_fc2


def inference(h_fc2, n_classes):
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


train_dir = 'D:\\大数据与人工智能实训\\course-of-big-data-and-AI-training\\5th\\flowers'
# train_dir = 'flowers'
logs_train_dir = 'CK-_part'
train, train_label = get_file(train_dir)
print('loaded')
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)
train_batch, train_label_batch = get_batch(x, y, IMG_W, IMG_H)
conv1 = conv1(train_batch)
pool1 = pool1(conv1)
conv2 = conv2(pool1)
pool2 = pool2(conv2)
conv3 = conv3(pool2)
pool3 = pool3(conv3)
fc1 = fc1(pool3, BATCH_SIZE)
fc2 = fc2(fc1)
train_logits = inference(fc2, N_CLASSES)
train_loss = losses(train_logits, train_label)
train_op = training(train_loss, learning_rate)
train_acc = evaluation(train_logits, train_label)

summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

if __name__ == "__main__":
    print('start')
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            a = train[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            b = train_label[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]

            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                            feed_dict={x: a, y: b})
            if step % 10 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
            saver.save(sess, checkpoint_path)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
    coord.join(threads)
    # sess.close()

    print("hello world")

    # ori image
    image_contents = tf.read_file(train[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    img = sess.run((image))
    print(img.shape)
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    ax2.imshow(img)

    # conv1

    # image = tf.image.resize_image_with_crop_or_pad(image, IMG_W, IMG_H)
    # image = tf.image.per_image_standardization(image)
    # image = tf.cast(image, tf.float32)
    # input_image = sess.run(image)
    # conv1_img = sess.run(conv1, feed_dict={images:input_image})
    conv1_img = sess.run(conv1)
    conv1_transpose = sess.run(tf.transpose(conv1_img, [3, 0, 1, 2]))
    # fig3, ax3 = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
    fig3, ax3 = plt.subplots(nrows=8, ncols=8, figsize=(28, 28))
    for i in range(8):
        for j in range(8):
            ax3[i][j].imshow(conv1_transpose[i][0])
    plt.title('Conv1 16x28x28')
    plt.show()
    #
    # conv1_img = sess.run(conv2)
    # conv1_transpose = sess.run(tf.transpose(conv1_img, [3, 0, 1, 2]))
    # # fig3, ax3 = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
    # fig3, ax3 = plt.subplots(nrows=8, ncols=8, figsize=(28, 28))
    # for i in range(8):
    #     for j in range(8):
    #         ax3[i][j].imshow(conv1_transpose[i][0])
    # plt.title('Conv1 16x28x28')
    # plt.show()

    # plt.show()
