import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def load_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_concat = tf.concat([mnist.train.images, mnist.validation.images], axis=0)
    y_concat = tf.concat([mnist.train.labels, mnist.validation.labels], axis=0)
    x_train = np.array([x for (x,y) in zip(x_concat.eval(), y_concat.eval()) if y < 5])
    y_train = tf.one_hot(tf.convert_to_tensor(np.array([y for y in y_concat.eval() if y < 5])), depth=5).eval()
    x_test = np.array([x for (x, y) in zip(mnist.test.images, mnist.test.labels) if y < 5])
    y_test = tf.one_hot(tf.convert_to_tensor(np.array([y for y in mnist.test.labels if y < 5])), depth=5).eval()
    # print(x_train) # Tensor("Const:0", shape=(30596, 784), dtype=float32)
    # print(y_train) # Tensor("one_hot:0", shape=(30596, 5), dtype=float32)
    # print(x_test) # Tensor("Const_2:0", shape=(5139, 784), dtype=float32)
    # print(y_test) # Tensor("Const_3:0", shape=(5139,), dtype=uint8)

    return x_train, y_train, x_test, y_test


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(train, labels, learning_rate, batch_size, steps, sess):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 5])
    # 1st layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Output layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.global_variables_initializer().run()

    i = 0
    for _ in range(steps):
        mini_batches = np.array_split(range(train.shape[0]), batch_size)
        for batch in mini_batches:
            print("step %d" % i)
            i += 1
            train_step.run(feed_dict={x: train[batch], y_: labels[batch]})

    return accuracy, x, y_


def test_accuracy(accuracy, test, labels, x, y_, sess):
    print(sess.run(accuracy, feed_dict={x: test, y_: labels}))


def main():
    sess = tf.InteractiveSession()
    x_train, y_train, x_test, y_test = load_data()
    accuracy, x, y_ = convolutional_neural_network(train=x_train, labels=y_train, learning_rate=1e-4, batch_size=1000, steps=10, sess=sess)
    test_accuracy(accuracy=accuracy, test=x_test, labels=y_test, x=x, y_=y_, sess=sess)
    sess.close()


main()
