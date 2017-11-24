"""
模型测试
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cnn_mnist_inference
import cnn_mnist_train
import numpy as np

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32,
            [mnist.validation.num_examples,
             cnn_mnist_inference.IMAGE_SIZE,
             cnn_mnist_inference.IMAGE_SIZE,
             cnn_mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, cnn_mnist_inference.OUTPUT_NODE], name='y-input')

        xs = mnist.validation.images
        reshaped_xs = np.reshape(xs,
                                 (mnist.validation.num_examples,
                                  cnn_mnist_inference.IMAGE_SIZE,
                                  cnn_mnist_inference.IMAGE_SIZE,
                                  cnn_mnist_inference.NUM_CHANNELS))

        validate_feed = {x: reshaped_xs,
                         y_: mnist.validation.labels}

        y = cnn_mnist_inference.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            cnn_mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(cnn_mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    for v in tf.global_variables():
                        print(v.name, ":", v.eval())
                    print("#####################")
                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]
                    accuracy_score = sess.run(
                        accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (
                        global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(
        "./datasets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
