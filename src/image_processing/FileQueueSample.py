import tensorflow as tf

######## 写入文件 ###############################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下将数据写入不同的文件
num_shards = 2  # 写入的文件数量
instances_per_shard = 2  # 每个文件中的数据量

for i in range(num_shards):
    filename = "datasets/tmp/data.tfrecords-%.5d-of-%.5d" % (i, num_shards)
    writer = tf.python_io.TFRecordWriter(filename)

    for j in range(instances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)
        }))
        writer.write(example.SerializeToString())

    writer.close()
######## 写入文件 ###############################################################


######## 读取文件 ###############################################################
# 获取文件列表
files = tf.train.match_filenames_once("./datasets/tmp/data.tfrecords-*")

# 创建输入队列, 这一个函数可以设置 num_epochs 参数来限制加载初始文件的最大轮数
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取文件并且解析数据
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }
)

with tf.Session() as sess:
    # 使用 match_filenames_once 需要初始化一些变量
    init = tf.global_variables_initializer(), tf.local_variables_initializer()
    sess.run(init)

    print(sess.run(files))

    # 启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
"""
result:
[0, 0]
[0, 1]
[1, 0]
[1, 1]
[0, 0]
[0, 1]
"""
######## 读取文件 ###############################################################


######## 组合训练数据 ####################################################
# 假设这里 example 表示一个向量特征，label 表示这张图片的标签
example, label = features['i'], features['j']

batch_size = 3

# 队列的容量
capacity = 100 + 3 * batch_size

# 把数据组织成 batch (添加多一维的数据)
exampel_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)
# 或者使用 shuffle_batch 函数进行乱序的组织, 
# min_after_dequeu 表示出队后至少保留的数据,
# nuum_threads 表示指定过个线程同时执行入队 (这个队列由 input_string_product 产生) 操作
exampel_batch, label_batch = tf.train.shuffle_batch(
    [exampel_batch, label], batch_size = batch_size,
    capacity=capacity, min_after_dequeue=30
)

with tf.Session() as sess:
    init = tf.global_variables_initializer(), tf.local_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [exampel_batch, label_batch]
        )
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join()
"""
result:
[0 0 1] [0 1 0]
[1 0 0] [1 0 1]
"""
######## 组合训练数据 ####################################################
