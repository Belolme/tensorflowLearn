import tensorflow as tf


ACTIONS = 9
REGULARIZER_RATE = 0.0001
LEARNING_RATE = 1e-4
DRAOPOUT_KEEP_PROB = 0.3


def __weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def __biasVariable(shape):
    # initial = tf.constant(0.01, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def __conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def __createNetworkWithCNN1(input_tensor, regularizer=None, train=False):
    # network weights
    with tf.variable_scope('first_layout'):
        w_conv1 = __weightVariable([3, 3, 1, 32])
        b_conv1 = __biasVariable([32])
        h_conv1 = tf.nn.leaky_relu(__conv2d(
            input_tensor, w_conv1, 1) + b_conv1)
        tf.summary.histogram("weights_1", w_conv1)
        tf.summary.histogram("biases_1", b_conv1)
        tf.summary.histogram("activations_1", h_conv1)

    with tf.variable_scope('second_layout'):
        w_conv2 = __weightVariable([3, 3, 32, 64])
        b_conv2 = __biasVariable([64])
        h_conv2 = tf.nn.leaky_relu(__conv2d(
            h_conv1, w_conv2, 1) + b_conv2)
        tf.summary.histogram("weights_2", w_conv2)
        tf.summary.histogram("biases_2", b_conv2)
        tf.summary.histogram("activations_2", h_conv2)

    with tf.variable_scope('third_layout'):
        w_conv3 = __weightVariable([3, 3, 64, 64])
        b_conv3 = __biasVariable([64])
        h_conv3 = tf.nn.leaky_relu(__conv2d(
            h_conv2, w_conv3, 1) + b_conv3)

    with tf.variable_scope('full_connect_layout'):
        w_fc1 = __weightVariable([576, 128])
        b_fc1 = __biasVariable([128])
        h_conv_flat = tf.reshape(h_conv3, [-1, 576])
        h_fc1 = tf.nn.tanh(tf.matmul(h_conv_flat, w_fc1) + b_fc1)

        if regularizer is not None:
            tf.add_to_collection('regularizer', regularizer(w_fc1))

        if train:
            w_fc1 = tf.nn.dropout(w_fc1, DRAOPOUT_KEEP_PROB)

        tf.summary.histogram("weights_3", w_fc1)
        tf.summary.histogram("biases_3", b_fc1)
        tf.summary.histogram("activations_3", h_fc1)

    with tf.variable_scope('output_layout'):
        w_fc2 = __weightVariable([128, ACTIONS])
        b_fc2 = __biasVariable([ACTIONS])
        y = tf.matmul(h_fc1, w_fc2) + b_fc2

        if regularizer is not None:
            tf.add_to_collection('regularizer', regularizer(w_fc2))

        tf.summary.histogram("weights_4", w_fc2)
        tf.summary.histogram("biases_4", b_fc2)
        tf.summary.histogram("activations_4", y)

    return y


def getCostFun(action, output_q, output_label):
    readout_action = tf.reduce_sum(tf.multiply(action, output_q), axis=1)
    cost = tf.reduce_mean(tf.square(readout_action - output_label)) + \
        tf.add_n(tf.get_collection('regularizer'))
    tf.summary.scalar('lost_value', cost)
    return cost


def getTrainStepAndLossFun(action, output_q, output_label):
    with tf.variable_scope('loss_function'):
        cost = getCostFun(action, output_q, output_label)

    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #     1e-2,
        #     global_step,
        #     5000, 0.96,
        #     staircase=True)
        # tf.summary.scalar('learning_rate', learning_rate)

        learning_rate = LEARNING_RATE
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cost, global_step=global_step)

    return train_step, cost


def createNetworkWithCNN1(input_tensor, train=False):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    return __createNetworkWithCNN1(input_tensor, regularizer, train)
