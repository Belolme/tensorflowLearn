import tensorflow as tf
from numpy.random import RandomState

batch_size = 10
dataset_size = 10000

# 这是边权矩阵的定义
w1 = tf.Variable([[1.87565053, -1.92562675, 2.03996181, -2.19880748],
                  [2.06988621, -2.07578921, 1.89732957, -1.79729867]], trainable=True)
w2 = tf.Variable([[-19.43003464],
                  [87.04914856],
                  [-18.81079865],
                  [89.67340088]], trainable=True)

# 这是输入节点和输出节点的定义
x_ = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 这是中间节点的定义
a = tf.sigmoid(tf.matmul(x_, w1))
b = tf.sigmoid(tf.matmul(a, w2))

# 这是 backprogation 算法的定义
cross_entropy = tf.reduce_mean(tf.square(y_ - tf.clip_by_value(b, 0, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 以下是输入集和输出集的定义
rdm = RandomState(1)
# X= [[x1 * 0.001, 1 - 0.001 * x1] for x1 in range(0, 1000)]
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1.0)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 1
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = (i * batch_size) % dataset_size + batch_size
        sess.run(train_step, feed_dict={x_: X[start: end], y_: Y[start: end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x_: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (
                i, total_cross_entropy))
            print("w1:", sess.run(w1))
            print("w2:", sess.run(w2))

            print(sess.run(tf.clip_by_value(b, 1e-10, 1.0),
                           feed_dict={x_: [[0.5465, 0.2111],
                                           [0.21, 0.78],
                                           [0.21, 0.79],
                                           [0.8, 0.9555],
                                           [0.8, 0.2],
                                           [0.5, 0.5],
                                           [0.3, 0.7],
                                           [1.0, 0.01]]}))
