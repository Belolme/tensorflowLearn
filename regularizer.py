import tensorflow as tf

def get_weight(shape, λ):
    """
    获取一层神经网络边上的权重

    Args:
        shape: 神经网络边权矩阵形状
        λ: regularizer 的 lambda 参数。此参数表示模型复杂损失在总损失中的比例。

    Return:
        返回一层神经网络的权重
    """
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    # 添加到 tensorflow 提供的集合当中，其中 losses 是这个 List 集合的名字
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(λ)(var))

    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 定义每一层网络中的节点个数和层数
layer_demension = [2, 10, 10, 10, 1]
n_layers = len(layer_demension)

cur_layer = x
in_deimension = layer_demension[0]

# 构建神经网络
for i in range(1, n_layers):
    out_dimension = layer_demension[i]
    weight = get_weight([in_deimension, out_dimension], .001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    cur_layer = tf.nn.relu(cur_layer @ weight + bias)

    in_deimension = out_dimension

# 构建拼装 coss function
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses', mes_loss)
loss = tf.add_n(tf.get_collection('losses'))
