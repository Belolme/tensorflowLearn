import tensorflow as tf
import numpy as np

ACTIONS = 9


def __weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def __biasVariable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def __conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def createNetwork(input_tensor):
    # network weights
    with tf.variable_scope('first_layout'):
        W_conv1 = __weightVariable([2, 2, 1, 32])
        b_conv1 = __biasVariable([32])
        h_conv1 = tf.nn.relu(__conv2d(
            input_tensor, W_conv1, 1) + b_conv1)

    with tf.variable_scope('second_layout'):
        W_conv2 = __weightVariable([2, 2, 32, 64])
        b_conv2 = __biasVariable([64])
        h_conv2 = tf.nn.relu(__conv2d(
            h_conv1, W_conv2, 1) + b_conv2)

    with tf.variable_scope('full_connect_layout'):
        W_fc1 = __weightVariable([64 * 9, 128])
        b_fc1 = __biasVariable([128])
        h_conv2_flat = tf.reshape(h_conv2, [-1, 64 * 9])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    with tf.variable_scope('output_layout'):
        W_fc2 = __weightVariable([128, ACTIONS])
        b_fc2 = __biasVariable([ACTIONS])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y

def getIndexWithOp(input_t, validate_t, validate_v, op):
    """
    get index in set of input_t and validate_t equal validate_v and op is True
    """
    result = -1
    for i, validate_value in enumerate(validate_t):
        print(i, validate_value)
        if validate_value == validate_v:
            if result == -1:
                result = i
            elif op(input_t[i], input_t[result]):
                result = i

    return result

def getMaxIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v,
                                        lambda x, y: x > y)

def getMinIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v,
                                        lambda x, y: x < y)

def main():
    with tf.Session() as sess:
        x = tf.placeholder('float', [None, 3, 3, 1])
        a = np.arange(0, 9).reshape([3, 3, 1])
        y = createNetwork(x)
        y2 = createNetwork(x)
        sess.run(tf.global_variables_initializer())
        out = sess.run(y, feed_dict={x: [a]})
        print(out)
        out = sess.run(y, feed_dict={x: [a]})
        print(out)

        out2 = sess.run(y2, feed_dict={x: [a]})
        print(out2)

        out = sess.run(y, feed_dict={x: [a]})
        print(out)

    # print(getMinIndex([1, 2, 3, 4], [0,1, 1, 0], 1))


if __name__ == '__main__':
    main()
