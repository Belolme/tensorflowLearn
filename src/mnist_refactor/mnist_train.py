"""
定义神经网络的训练过程
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train(mnist):
    """
    定义一个训练模型
    """
    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # config EMA
    # with tf.name_scope('moving_average'):
    #     variable_averages = tf.train.ExponentialMovingAverage(
    #         MOVING_AVERAGE_DECAY, global_step)
    #     variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # cofig coss function
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        # 可视化 loss_entropy
        tf.summary.scalar('loss', loss)

    # config learning rate
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)

        # construct train step
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)

        # with tf.control_dependencies([train_step, variables_averages_op]):
        #     train_op = tf.no_op(name='train')

    # config saver
    with tf.name_scope('save_and_tfb'):
        saver = tf.train.Saver()
        summ = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 将当前的计算图输出到 TensorBoard 日志文件
        writer = tf.summary.FileWriter('./datasets/log', tf.get_default_graph())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, summ_output = sess.run(
                [train_step, loss, global_step, summ], feed_dict={x: xs, y_: ys})
            writer.add_summary(summ_output, i)

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (
                    step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                              MODEL_NAME), global_step=global_step)

        writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets(
        "./datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
