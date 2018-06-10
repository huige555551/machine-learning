import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def load_data():
    mnist = input_data.read_data_sets("../logistic-regression/MNIST_data/", one_hot=False)
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


def convolutional_neural_network(train, labels, learning_rate, sess):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 5])
    W = tf.Variable(tf.zeros([784, 5]))
    b = tf.Variable(tf.zeros([5]))

    tf.global_variables_initializer().run()

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    for i in range(100):
        mini_batches = np.array_split(range(train.shape[0]), 50)
        for batch in mini_batches:
            train_step.run(feed_dict={x: train[batch], y_: labels[batch]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy, x, y_


def test_accuracy(accuracy, test, labels, x, y_, sess):
    print(sess.run(accuracy, feed_dict={x: test, y_: labels}))


def main():
    sess = tf.InteractiveSession()
    x_train, y_train, x_test, y_test = load_data()
    accuracy, x, y_ = convolutional_neural_network(train=x_train, labels=y_train, learning_rate=0.5, sess=sess)
    test_accuracy(accuracy=accuracy, test=x_test, labels=y_test, x=x, y_=y_, sess=sess)
    sess.close()


main()
