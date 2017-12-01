import tensorflow as tf

input1 = tf.constant([1., 2., 3.], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

# use tensorflow#train#SummaryWriter create log file
writer = tf.summary.FileWriter("./datasets/log", tf.get_default_graph())
writer.close()
