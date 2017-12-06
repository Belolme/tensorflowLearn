import tensorflow as tf
from numpy.random import RandomState
import numpy as np

BATCH_SIZE = 5000
DATASET_SIZE = 20000

# 这是边权矩阵的定义
w1 = tf.Variable(tf.random_normal([2, 4], stddev=0.1), trainable=True)
w2 = tf.Variable(tf.random_normal([4, 16]), trainable=True)
w3 = tf.Variable(tf.random_normal([16, 2]), trainable=True)

# 这是输入节点和输出节点的定义
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 2), name='y-input')

# 这是中间节点的定义
a = tf.nn.sigmoid(tf.matmul(x, w1))
b = tf.nn.sigmoid(a @ w2)
y = tf.matmul(b, w3)
tf.summary.histogram("w1", w1)
tf.summary.histogram("w2", w2)
tf.summary.histogram("w3", w3)

# 这是 backprogation 算法的定义
# cross_entropy = tf.reduce_mean(tf.square(y_ - tf.clip_by_value(y, 0, 1.0)))
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.nn.softmax(y)))

# use tensorboard visual cross_entropy
tf.summary.scalar('cross_entropy', cross_entropy)
summ = tf.summary.merge_all()

# 设置指数衰减的学习率
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    0.001,
    global_step,
    1000,
    0.96,
    staircase=True)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

# 以下是输入集和输出集的定义
rdm = RandomState(1)
X = rdm.rand(DATASET_SIZE - 3000, 2)
X_extern = [[x1 * 0.001, 1 - 0.001 * x1] for x1 in range(0, 1000)]
x_extern2 = [[x1 * 0.001, 1 - 0.001 * x1 + .002] for x1 in range(0, 1000)]
x_extern3 = [[x1 * 0.001, 1 - 0.001 * x1 - .001] for x1 in range(0, 1000)]
X = np.row_stack((X, X_extern, x_extern2, x_extern3))

Y = [[float(x1 + x2 < 1.0), float(x1 + x2 >= 1.0)] for (x1, x2) in X]

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 输出目前（未经训练）的参数取值
    # print("w1:", sess.run(w1))
    # print("w3:", sess.run(w3))
    print("\n")
    writer = tf.summary.FileWriter('./datasets/log', tf.get_default_graph())

    # 训练模型。
    STEPS = 1000000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % DATASET_SIZE
        end = (i * BATCH_SIZE) % DATASET_SIZE + BATCH_SIZE
        _, summ_out, step = sess.run([train_step, summ, global_step], feed_dict={x: X[start: end], y_: Y[start: end]})
        writer.add_summary(summ_out, i)

        if i % 10000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (
                step, total_cross_entropy))

            print(sess.run(tf.clip_by_value(y, 1e-10, 1.0),
                           feed_dict={x: [[0.5465, 0.2111],
                           [0.21, 0.78],
                           [0.21, 0.79],
                           [0.8, 0.955],
                           [0.8, 0.2],
                           [0.5, 0.5],
                           [0.3, 0.7],
                           [1.0, 0.01]]}))

    writer.close()
