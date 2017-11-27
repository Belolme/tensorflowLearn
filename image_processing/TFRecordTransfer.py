import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_List=tf.train.Int64List(value=[value]))

def _byte_feature(value):
    return tf.train.Feature(bytes_List=tf.train.BytesList(value=[value]))

# 下面生成的 mnist 是一个 class，里面的属性有 train, validation, test
mnist = input_data.read_data_sets("./datasets/MNIST_data", dtype=tf.unit8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

# 训练数据的图像分辨率
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出 TFRecord 文件的地址
filename = "/datasets/MNIST_data/tfrecord_output"

# 创建一个 writer 来写 TFRecord 文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(feature=tf.train.Feature(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])), # 返回最大值的下标[http://blog.csdn.net/oHongHong/article/details/72772459]
        'image_raw': _byte_feature(image_raw)
    }))

    # 将一个 Example 写入 TFRecord 文件
    writer.write(example.SerializeToString())
writer.close();

# 读取 TFRecord 文件的示例
reader = tf.TFRecordReader()

# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer([filename])

# 从文件中读出一个样例，也可以使用 read_up_to 函数一次性读取多个样例
_, serialized_example = reader.read(filename_queue)

# 解析出 features (内含多个 feature), 如果需要解析出多个 example 可以使用 tf#parse_example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image_raw': tf.FixedLenFeature([], tf.string),
                                       'pixels': tf.FixedLenFeature([], tf.int64),
                                       'label': tf.FixedLenFeature([], tf.int64)
                                   })

# 解析 feature
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 多线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


