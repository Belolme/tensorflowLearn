"""
定义前向传播的过程以及神经网络中的参数
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1

# 卷积层过滤器尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

# 全连接层节点个数
FC_SIZE = 512

# 输出节点个数
NUM_LABELS = 10


def inference(input_tensor, train, regularizer):
    """
    构建神经网络的结构
    """
    # 第一层卷积1
    # 输入为[x-size=28, y-size=28, channel=1]的图像
    # 过滤器尺寸[x-size=5, y-size=5, channel=1, deep=32]
    # 过滤器步长=1
    # 输出为[x-size=28, y-size=28, deep=32]的矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            name="weight",
            shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            name="bias",
            shape=[CONV1_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化1
    # 输入为[x-size=28, y-size=28, deep=32]的矩阵
    # 过滤器尺寸[x-size=2, y-size=2]
    # 过滤器步长=2
    # 输出为[x-size=14, y-size=14, deep=32]的矩阵
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding="SAME")

    # 第三层卷积2
    # 输入为[x-size=14, y-size=14, deep=32]的矩阵
    # 过滤器尺寸[x-size=5, y-size=5, channel=1, deep=64]
    # 过滤器步长=1
    # 输出为[x-size=14, y-size=14, deep=64]的矩阵
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            name="weight",
            shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            name="bias",
            shape=[CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
                             1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化2
    # 输入为[x-size=14, y-size=14, deep=64]的矩阵
    # 过滤器尺寸[x-size=2, y-size=2]
    # 过滤器步长=2
    # 输出为[x-size=7, y-size=7, deep=64]的矩阵
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

    # 把[batch, x-size, y-size, deep]4维矩阵转化为[batch, vector]2维矩阵
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 全连接层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            name="weight",
            shape=[nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 只有全连接的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # dropout在训练数据的时候，会随机把部分输出改为0
        # dropout可以避免过度拟合，dropout一般只在全连接层，而不是在卷积层或者池化层使用
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 全连接层
    # 输入为[512]的向量
    # 输出为[10]的向量
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            name="weight",
            shape=[FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
