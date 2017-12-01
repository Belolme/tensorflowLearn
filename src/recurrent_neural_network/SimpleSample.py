"""
RNN 前向传播 sample
"""

import numpy as np


# 输入数据
X = [1, 2]
state = [0.0, 0.0]  # 初始状态输入

# 中间转换矩阵
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])  # baise

# 输出转换矩阵
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程
for i, _ in enumerate(X):
    # 计算循环体中的全连接层神经网络
    # np#doc 表示矩阵乘法
    before_activation = np.dot(state, w_cell_state) + \
        np.dot(X[i], w_cell_input) + b_cell

    state = np.tanh(before_activation)

    final_output = np.dot(state, w_output) + b_output

    print("before activation: ", before_activation)
    print("state: ", state)
    print("output: ", final_output)
