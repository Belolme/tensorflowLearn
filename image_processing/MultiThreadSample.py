import tensorflow as tf
import numpy as np
import threading
import time

################### queue smaple ######################
# 创建一个 FIFO 队列，队列最多可以保存两个元素，类型为 int32
q = tf.FIFOQueue(2, "int32")

# 入队（和变量一样，队列也需要初始化操作）
init = q.enqueue_many([[0, 10]])

x = q.dequeue()

y = x + 1

q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 初始化队列
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)
"""
result: 
0
10
1
11
2
"""
################### queue smaple ######################


################### Coorinator sample #################
def MyLoop(coord, worker_id):
    """
    这一个方法是线程的执行体，相当于 java 中 Runnable#run 方法
    """
    # 这里是手动调用 coord#should_stop 进行判断
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stop from id: %d\n" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id)

        time.sleep(1)

coord = tf.train.Coordinator()

# 创建 5 个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]

for t in threads: t.start()

coord.join(threads)
"""
result:
Working on id: 0
Working on id: 1
Working on id: 2
Working on id: 3
Stop from id: 4
"""
################### Coorinator sample #################


################### QueueRunner smaple ################
# 定义一个 100 个元素的 queue
queue = tf.FIFOQueue(100, 'float')

# 入队操作的定义, 随机添加一个一维 tensor
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 同一时间进行 5 次入队操作（开启 5 个线程）
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将 queueRunner 加入 TensorFlow 计算图中指定的集合
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    # 启动 QueueRunner （多线程），这一个操作将会启动所有线程的入队操作，
    # 当调用出队操作时，程序会一直等待入队操作被运行
    tf.train.start_queue_runners(sess=sess, coord=coord)

    # 进行 3 次出队操作，每一出队操作将会调用运行入队操作。
    # 根据上面的定义，每一次入队操作将会有 5 个 tensor 被入队。
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    coord.request_stop()
    coord.join(threads)
"""
上面程序将启动 5 个线程来进行队列的入队操作，每一个线程都是将随机 tensor 写入队列。
result:
-0.332199
1.52843
-0.0961194
"""
################### QueueRunner smaple ################
