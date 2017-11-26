"""
定义神经网络的训练过程
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cnn_mnist_inference
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./datasets/MNIST_model/"
MODEL_NAME = "cnn_mnist_model"


def train(mnist):
    """
    定义一个训练模型
    """
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        cnn_mnist_inference.IMAGE_SIZE,
                        cnn_mnist_inference.IMAGE_SIZE,
                        cnn_mnist_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [BATCH_SIZE, cnn_mnist_inference.OUTPUT_NODE],
                        name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = cnn_mnist_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # config EMA
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # cofig coss function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # config learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    # construct train step
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # config saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,
                                     (BATCH_SIZE,
                                      cnn_mnist_inference.IMAGE_SIZE,
                                      cnn_mnist_inference.IMAGE_SIZE,
                                      cnn_mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (
                    step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(
        "./datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
