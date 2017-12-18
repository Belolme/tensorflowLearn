"""
tic tac toe game ai
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
import game.TicTacToeGame as game

ACTIONS = 9  # number of valid actions
GAMMA = .99  # decay rate of past observations
OBSERVE = 128.  # timesteps to observe before training
EXPLORE = 200.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 129  # number of previous transitions to remember
BATCH = 128  # size of minibatch
TRAIN_STEP = 100


class DeepQNetwork:

    def __weightVariable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def __biasVariable(shape):
        # initial = tf.constant(0.01, shape=shape)
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def __conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def createNetworkWithCNN1(self, input_tensor):
        # network weights
        with tf.variable_scope('first_layout'):
            W_conv1 = DeepQNetwork.__weightVariable([3, 3, 1, 32])
            b_conv1 = DeepQNetwork.__biasVariable([32])
            h_conv1 = tf.nn.leaky_relu(DeepQNetwork.__conv2d(
                input_tensor, W_conv1, 1) + b_conv1)
            tf.summary.histogram("weights_1", W_conv1)
            tf.summary.histogram("biases_1", b_conv1)
            tf.summary.histogram("activations_1", h_conv1)

        # with tf.variable_scope('second_layout'):
        #     W_conv2 = DeepQNetwork.__weightVariable([3, 3, 32, 64])
        #     b_conv2 = DeepQNetwork.__biasVariable([64])
        #     h_conv2 = tf.nn.leaky_relu(DeepQNetwork.__conv2d(
        #         h_conv1, W_conv2, 1) + b_conv2)
        #     tf.summary.histogram("weights_2", W_conv2)
        #     tf.summary.histogram("biases_2", b_conv2)
        #     tf.summary.histogram("activations_2", h_conv2)

        # with tf.variable_scope('third_layout'):
        #     w_conv3 = DeepQNetwork.__weightVariable([3, 3, 64, 64])
        #     b_conv3 = DeepQNetwork.__biasVariable([64])
        #     h_conv3 = tf.nn.leaky_relu(DeepQNetwork.__conv2d(
        #     h_conv2, w_conv3, 1) + b_conv3)

        with tf.variable_scope('full_connect_layout'):
            W_fc1 = DeepQNetwork.__weightVariable([32, 128])
            b_fc1 = DeepQNetwork.__biasVariable([128])
            h_conv_flat = tf.reshape(h_conv1, [-1, 32])
            h_fc1 = tf.nn.tanh(tf.matmul(h_conv_flat, W_fc1) + b_fc1)
            tf.summary.histogram("weights_3", W_fc1)
            tf.summary.histogram("biases_3", b_fc1)
            tf.summary.histogram("activations_3", h_fc1)

        with tf.variable_scope('output_layout'):
            W_fc2 = DeepQNetwork.__weightVariable([128, ACTIONS])
            b_fc2 = DeepQNetwork.__biasVariable([ACTIONS])
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            tf.summary.histogram("weights_4", W_fc2)
            tf.summary.histogram("biases_4", b_fc2)
            tf.summary.histogram("activations_4", y)

        return y

    def createNetworkWithCNN2(self, input_tensor):
        """
        构建神经网络的结构
        """
        # 第一层卷积1
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                name="weight",
                shape=[2, 2, 1, 32],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            conv1_biases = tf.get_variable(
                name="bias",
                shape=[32],
                initializer=tf.constant_initializer(0.0)
            )
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                                strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        # 第二层池化1
        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")

        # 第三层卷积2
        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable(
                name="weight",
                shape=[2, 2, 32, 64],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            conv2_biases = tf.get_variable(
                name="bias",
                shape=[64],
                initializer=tf.constant_initializer(0.0)
            )
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
                1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        # 把[batch, x-size, y-size, deep]4维矩阵转化为[batch, vector]2维矩阵
        pool_shape = relu2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(relu2, [-1, nodes])

        # 全连接层
        with tf.variable_scope('layer5-fc1'):
            fc1_weights = tf.get_variable(
                name="weight",
                shape=[nodes, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )

            fc1_biases = tf.get_variable(
                "bias", [512], initializer=tf.constant_initializer(0.1))

            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable(
                name="weight",
                shape=[512, ACTIONS],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            fc2_biases = tf.get_variable(
                "bias", [ACTIONS], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        return logit

    def createFCNetwork(input_tensor):
        # 这是边权矩阵的定义
        w1 = tf.Variable(tf.random_normal([9, 32], stddev=0.1), trainable=True)
        w2 = tf.Variable(tf.random_normal([32, 64]), trainable=True)
        w3 = tf.Variable(tf.random_normal([64, 9]), trainable=True)

        # 这是中间节点的定义
        a = tf.nn.sigmoid(tf.matmul(input_tensor, w1))
        b = tf.nn.sigmoid(a @ w2)
        y = tf.matmul(b, w3)

        return y

    def getCostFun(self, action, output_q, output_label):
        # readout_action = tf.reduce_sum(tf.multiply(action, output_q), axis=1)
        readout_action = output_q
        cost = tf.reduce_mean(tf.square(output_label - readout_action))
        tf.summary.scalar('lost_value', cost)
        return cost

    def getTrainStepAndLossFun(self, action, output_q, output_label):
        with tf.variable_scope('loss_function'):
            cost = self.getCostFun(action, output_q, output_label)

        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                0.1,
                global_step,
                5000, 0.96,
                staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            # learning_rate = 1e-2
            train_step = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(cost, global_step=global_step)
        
        return train_step, cost

    def train(self):
        # input layer
        with tf.variable_scope('input_tensor'):
            input_tensor = tf.placeholder("float", [None, 3, 3, 1])
            output_label = tf.placeholder('float', [None, 9])
            action = tf.placeholder('float', [None, ACTIONS])

        output_q = self.createNetworkWithCNN1(input_tensor)

        train_step, loss = self.getTrainStepAndLossFun(action, output_q, output_label)
        summ = tf.summary.merge_all()

        with tf.Session() as sess:
            # saving and loading networks
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state("saved_networks")
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
                print("Could not find old network weights")

            # init game
            my_game = game.TicTacToe()

            # start training
            epsilon = INITIAL_EPSILON
            times = 0
            D = deque()

            writer = tf.summary.FileWriter('./datasets/log', tf.get_default_graph())

            # 黑子取 max_q, 白子取 min_q
            while True:
                # 重新开局
                if my_game.terminal != game.TerminalStatus.GOING:
                    my_game.reset()

                state = my_game.state.copy()
                is_turn_to = my_game.is_turn

                # get current q value
                next_q = sess.run(output_q,
                                feed_dict={input_tensor: [state.reshape([3, 3, 1])]})[0]

                # get action index
                action_index = -1
                action_tensor = np.zeros([ACTIONS])
                if random.random() < epsilon:
                    print('-------random action----------')
                    random_set = []
                    for i, v in enumerate(state.reshape([-1])):
                        if v == game.IsTurnTo.BLANK.value:
                            random_set.append(i)

                    action_index = random.sample(random_set, 1)[0]
                else:
                    print('------------q value action--------------')
                    if game.IsTurnTo.BLACK == is_turn_to:
                        action_index = DeepQNetwork.getMaxIndex(next_q, state.reshape([-1]),
                                                                game.IsTurnTo.BLANK.value)
                    else:
                        action_index = DeepQNetwork.getMinIndex(next_q, state.reshape([-1]),
                                                                game.IsTurnTo.BLANK.value)

                action_tensor[action_index] = 1
                choice_q_value = next_q[action_index]

                # scale down epsilon
                if epsilon > FINAL_EPSILON and len(D) > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                print(state)
                action_position = (int(action_index / game.CHESSBOARD_SIZE),
                                int(action_index % game.CHESSBOARD_SIZE))
                print("action", action_position)
                my_game.setAction(action_position)
                next_state, reward, terminal = my_game.getState()

                new_m = (state, is_turn_to, action_tensor.copy(), reward, next_state.copy(), terminal)
                if len(D) > REPLAY_MEMORY:
                    D.popleft()
                for d in D:
                    if np.array_equal(new_m[0], d[0]) and np.array_equal(new_m[2], d[2]):
                        break
                else:
                    D.append(new_m)
                print('D lenght is: ',len(D))
                print('new_m is ', new_m)

                if len(D) > OBSERVE:
                    def getOutputQLabel(batch):
                        side_batch = [d[1] for d in batch]
                        reward_batch = [d[3] for d in batch]
                        next_state_batch = [d[4].reshape([3,3,1]) for d in batch]

                        return_batch = []
                        for i in range(0, len(batch)):
                            terminal = batch[i][5]

                            q_value = sess.run(output_q, feed_dict={input_tensor: [batch[i][0].reshape([3, 3, 1])]})[0]

                            if terminal != game.TerminalStatus.GOING:
                                q_value[np.argmax(batch[i][2])] = reward_batch[i]
                                return_batch.append(q_value)
                            else:
                                if side_batch[i] == game.IsTurnTo.BLACK:
                                    next_q_1 = sess.run(output_q, feed_dict={input_tensor: [next_state_batch[i]]})[0]
                                    next_q_value = next_q_1[DeepQNetwork.getMinIndex(next_q_1,
                                                                            next_state_batch[i].reshape([-1]),
                                                                            game.IsTurnTo.BLANK.value)]
                                else:
                                    next_q_1 = sess.run(output_q, feed_dict={input_tensor: [next_state_batch[i]]})[0]
                                    next_q_value = next_q_1[DeepQNetwork.getMaxIndex(next_q_1,
                                                                            next_state_batch[i].reshape([-1]),
                                                                            game.IsTurnTo.BLANK.value)]
                                q_value[np.argmax(batch[i][2])] = reward_batch[i] + GAMMA * next_q_value
                                return_batch.append(q_value)

                        return return_batch

                    # sample a minibatch to train on
                    minibatch = random.sample(D, BATCH)
                    # minibatch = []
                    # for i in range(0, BATCH):
                        # minibatch.append(D[-i])

                    while True:
                        output_q_batch = getOutputQLabel(minibatch)
                        # print(output_q_b_batch)

                        _, loss_output, summ_output = sess.run([train_step, loss, summ], feed_dict={action: [d[2] for d in minibatch],
                                                            input_tensor: [d[0].reshape([3, 3, 1]) for d in minibatch],
                                                            output_label: output_q_batch})
                        
                        writer.add_summary(summ_output, times)

                        if loss_output < 0.1:
                            break

                    if loss_output < 0.1 and epsilon <= FINAL_EPSILON:
                        epsilon = 0.1

                    times += 1

                    # save progress every 10000 iterations
                    if times % 1000 == 0:
                        saver.save(sess, "saved_networks/tictactoe-dqn",
                                global_step=times)

                    # print info
                    state = ""
                    if times <= OBSERVE:
                        state = "observe"
                    elif times > OBSERVE and times <= OBSERVE + EXPLORE:
                        state = "explore"
                    else:
                        state = "train"

                    print(terminal)
                    print("TIMESTEP", times, "/ STATE", state,
                        "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward,
                        "/ Q_Choice %s" % choice_q_value)
                    # print('loss output', loss_output)
                    print(next_q)

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
        return DeepQNetwork.getIndexWithOp(input_t, validate_t, validate_v,
                                           lambda x, y: x > y)

    def getMinIndex(input_t, validate_t, validate_v):
        return DeepQNetwork.getIndexWithOp(input_t, validate_t, validate_v,
                                           lambda x, y: x < y)


if __name__ == "__main__":
    DeepQNetwork().train()
