import tensorflow as tf
import numpy as np
import game

ACTIONS = 9


def __weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    # initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)


def __biasVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    # initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)


def __conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def createNetwork(input_tensor):
    # network weights
    with tf.variable_scope('first_layout'):
        w_conv1 = __weightVariable([3, 3, 1, 32])
        b_conv1 = __biasVariable([32])
        h_conv1 = tf.nn.relu(__conv2d(input_tensor, w_conv1, 1) + b_conv1)

    with tf.variable_scope('second_layout'):
        w_conv2 = __weightVariable([3, 3, 32, 64])
        b_conv2 = __biasVariable([64])
        h_conv2 = tf.nn.relu(__conv2d(h_conv1, w_conv2, 1) + b_conv2)

    with tf.variable_scope('third_layout'):
        w_conv3 = __weightVariable([3, 3, 64, 64])
        b_conv3 = __biasVariable([64])
        h_conv3 = tf.nn.relu(__conv2d(h_conv2, w_conv3, 1) + b_conv3)

    with tf.variable_scope('full_connect_layout'):
        w_fc1 = __weightVariable([32 * 18, 128])
        b_fc1 = __biasVariable([128])
        h_conv_flat = tf.reshape(h_conv3, [-1, 32 * 18])
        h_fc1 = tf.nn.tanh(tf.matmul(h_conv_flat, w_fc1) + b_fc1)

    with tf.variable_scope('output_layout'):
        w_fc2 = __weightVariable([128, ACTIONS])
        b_fc2 = __biasVariable([ACTIONS])
        y = tf.matmul(h_fc1, w_fc2) + b_fc2

    return y


def createNetwork2(input_tensor):
    # 这是边权矩阵的定义
    w1 = tf.Variable(tf.random_normal([9, 32], stddev=0.1), trainable=True)
    w2 = tf.Variable(tf.random_normal([32, 64]), trainable=True)
    w3 = tf.Variable(tf.random_normal([64, 9]), trainable=True)

    # 这是中间节点的定义
    a = tf.nn.sigmoid(tf.matmul(input_tensor, w1))
    b = tf.nn.sigmoid(a @ w2)
    y = tf.matmul(b, w3)

    return y


def getCostFun(action, output_q, output_label):
    readout_action = tf.reduce_sum(tf.multiply(action, output_q), axis=1)
    cost = tf.reduce_mean(tf.square(readout_action - output_label))
    tf.summary.scalar('lost_value', cost)
    return cost
    

def getTrainStepAndLossFun(action, output_q, output_label):
    with tf.variable_scope('loss_function'):
        cost = getCostFun(action, output_q, output_label)

    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #     0.01,
        #     global_step,
        #     100, 0.96,
        #     staircase=True)
        # tf.summary.scalar('learning_rate', learning_rate)

        learning_rate = 1e-2
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cost, global_step=global_step)

    return train_step, cost


def getIndexWithOp(input_t, validate_t, validate_v, op):
    """
    get index in set of input_t and validate_t equal validate_v and op is True
    """
    result = -1
    for i, validate_value in enumerate(validate_t):
        if validate_value == validate_v:
            if result == -1:
                result = i
            elif op(input_t[i], input_t[result]):
                result = i

    return result


def getMaxIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v,
                                        lambda x, y: x > y)


def getMinIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v,
                                        lambda x, y: x < y)


def getOutputQLabel(batch, sess, y, input_tensor):
    turn_to_batch = [d[1] for d in batch]
    reward_batch = [d[3] for d in batch]
    next_state_batch = [d[4].reshape([3, 3, 1]) for d in batch]

    label_batch = []
    for i in range(0, len(batch)):
        terminal = batch[i][5]

        if terminal != game.TerminalStatus.GOING:
            label_batch.append(reward_batch[i])
        else:
            next_q_t = sess.run(y, feed_dict={input_tensor: [next_state_batch[i]]})[0]
            if turn_to_batch[i] == game.IsTurnTo.BLACK:
                next_q_value = next_q_t[getMinIndex(next_q_t,
                                            next_state_batch[i].reshape([-1]),
                                            game.IsTurnTo.BLANK.value)]
                next_q_value = -1 if next_q_value < -1 else next_q_value
            else:
                next_q_value = next_q_t[getMaxIndex(next_q_t,
                                            next_state_batch[i].reshape([-1]),
                                            game.IsTurnTo.BLANK.value)]
                next_q_value = 1 if next_q_value > 1 else next_q_value

            # print(next_q_t)
            # print(next_q_value)
            label_batch.append(next_q_value)

    return label_batch


def main():
    with tf.Session() as sess:
        # input_tensor = tf.placeholder("float", [None, 9])
        input_tensor = tf.placeholder("float", [None, 3, 3, 1])
        output_label = tf.placeholder('float', [None])
        action = tf.placeholder('float', [None, ACTIONS])

        y = createNetwork(input_tensor)
        train_step, loss = getTrainStepAndLossFun(action, y, output_label)

        sess.run(tf.initialize_all_variables())

        minibatch = [
            # (np.array([[1, 2, 1],[1, 1, 2],[0, 2, 2]]), game.IsTurnTo.BLACK, np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]), 10,
            #     np.array([[1, 2, 1],[1, 1, 2],[2, 2, 2]]), game.TerminalStatus.BLACK_WIN, 6),
            # (np.array([[1, 0, 1],[2, 0, 2],[0, 0, 2]]), game.IsTurnTo.WHITE, np.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), -10,
            #     np.array([[1, 1, 1],[2, 0, 2],[0, 0, 2]]), game.TerminalStatus.WHITE_WIN, 1),
            # (np.array([[2, 0, 0],[1, 1, 0],[2, 0, 0]]), game.IsTurnTo.BLACK, np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]), 0,
            #     np.array([[2, 0, 0],[1, 1, 0],[2, 2, 0]]), game.TerminalStatus.GOING, 7),
            # (np.array([[2, 0, 0],[1, 1, 0],[2, 2, 0]]), game.IsTurnTo.WHITE, np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]), -10,
            #     np.array([[2, 0, 0],[1, 1, 1],[2, 2, 0]]), game.TerminalStatus.WHITE_WIN, 5),
            # (np.array([[0, 2, 1],[2, 1, 0],[2, 0, 0]]), game.IsTurnTo.WHITE, np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]), 0,
            #     np.array([[0, 2, 1],[2, 1, 1],[2, 0, 0]]), game.TerminalStatus.GOING, 5),
            # (np.array([[0, 2, 1],[2, 1, 1],[2, 0, 0]]), game.IsTurnTo.BLACK, np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), 10,
            #     np.array([[2, 2, 1],[2, 1, 1],[2, 0, 0]]), game.TerminalStatus.BLACK_WIN, 0),

            (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 0, np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]), game.TerminalStatus.GOING, 4), 
            (np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]), game.IsTurnTo.WHITE, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]]), game.TerminalStatus.GOING, 0), 
            (np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 0, np.array([[1, 0, 0], [0, 2, 0], [2, 0, 0]]), game.TerminalStatus.GOING, 6),
            (np.array([[1, 0, 0], [0, 2, 0], [2, 0, 0]]), game.IsTurnTo.WHITE, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0, np.array([[1, 1, 0], [0, 2, 0], [2, 0, 0]]), game.TerminalStatus.GOING, 1),
            (np.array([[1, 1, 0], [0, 2, 0], [2, 0, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 0, np.array([[1, 1, 0], [0, 2, 0], [2, 0, 2]]), game.TerminalStatus.GOING, 8), 
            (np.array([[1, 1, 0], [0, 2, 0], [2, 0, 2]]), game.IsTurnTo.WHITE, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 0, np.array([[1, 1, 0], [0, 2, 0], [2, 1, 2]]), game.TerminalStatus.GOING, 7), 
            (np.array([[1, 1, 0], [0, 2, 0], [2, 1, 2]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1, np.array([[1, 1, 2], [0, 2, 0], [2, 1, 2]]), game.TerminalStatus.BLACK_WIN, 2),


            (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), game.IsTurnTo.BLACK, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),0, np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]), game.TerminalStatus.GOING,0), 
            (np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]), game.IsTurnTo.WHITE, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),0, np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]]), game.TerminalStatus.GOING,1), 
            (np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),0, np.array([[2, 1, 0], [2, 0, 0], [0, 0, 0]]), game.TerminalStatus.GOING,3), 
            (np.array([[2, 1, 0], [2, 0, 0], [0, 0, 0]]), game.IsTurnTo.WHITE, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),0, np.array([[2, 1, 0], [2, 1, 0], [0, 0, 0]]), game.TerminalStatus.GOING,4), 
            (np.array([[2, 1, 0], [2, 1, 0], [0, 0, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),0, np.array([[2, 1, 0], [2, 1, 0], [0, 2, 0]]), game.TerminalStatus.GOING,7), 
            (np.array([[2, 1, 0], [2, 1, 0], [0, 2, 0]]), game.IsTurnTo.WHITE, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),0, np.array([[2, 1, 0], [2, 1, 0], [1, 2, 0]]), game.TerminalStatus.GOING,6), 
            (np.array([[2, 1, 0], [2, 1, 0], [1, 2, 0]]), game.IsTurnTo.BLACK, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),0, np.array([[2, 1, 0], [2, 1, 0], [1, 2, 2]]), game.TerminalStatus.GOING,8), 
            (np.array([[2, 1, 0], [2, 1, 0], [1, 2, 2]]), game.IsTurnTo.WHITE, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),-1, np.array([[2, 1, 1], [2, 1, 0], [1, 2, 2]]), game.TerminalStatus.WHITE_WIN,2)
        ]

      
        times = 0
        for _ in range(0, 100000):
            output_q_batch = getOutputQLabel(minibatch, sess, y, input_tensor)
            # print(output_q_batch)
            _, loss_result = sess.run([train_step,loss],  feed_dict={action: [d[2] for d in minibatch],
                                    input_tensor: [d[0].reshape([3, 3, 1]) for d in minibatch],
                                    output_label: output_q_batch})
            
            times += 1
            print('lose result:', loss_result, 'times: ', times)

            if times % 100 == 0:
                state1 = minibatch[0][0].reshape([1, 3, 3, 1])
                # print('state1', state1)
                print(sess.run(y, feed_dict={input_tensor: state1}))

                state2 = minibatch[6][0].reshape([1, 3, 3, 1])
                # print('state 2: ', state2)
                print(sess.run(y, feed_dict={input_tensor:state2}))

if __name__ == '__main__':
    main()
