"""
tic tac toe game ai
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
import game.TicTacToeGame as game

ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 3000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.5  # starting value of epsilon
REPLAY_MEMORY = 5000  # number of previous transitions to remember
BATCH = 32  # size of minibatch


class DeepQNetwork:

    def __weightVariable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def __biasVariable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def __conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def createNetwork(self, input_tensor):
        # network weights
        with tf.variable_scope('first_layout'):
            W_conv1 = DeepQNetwork.__weightVariable([2, 2, 1, 32])
            b_conv1 = DeepQNetwork.__biasVariable([32])
            h_conv1 = tf.nn.relu(DeepQNetwork.__conv2d(
                input_tensor, W_conv1, 1) + b_conv1)

        with tf.variable_scope('second_layout'):
            W_conv2 = DeepQNetwork.__weightVariable([2, 2, 32, 64])
            b_conv2 = DeepQNetwork.__biasVariable([64])
            h_conv2 = tf.nn.relu(DeepQNetwork.__conv2d(
                h_conv1, W_conv2, 1) + b_conv2)

        with tf.variable_scope('full_connect_layout'):
            W_fc1 = DeepQNetwork.__weightVariable([576, 128])
            b_fc1 = DeepQNetwork.__biasVariable([128])
            h_conv2_flat = tf.reshape(h_conv2, [-1, 576])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        with tf.variable_scope('output_layout'):
            W_fc2 = DeepQNetwork.__weightVariable([128, ACTIONS])
            b_fc2 = DeepQNetwork.__biasVariable([ACTIONS])
            y = tf.matmul(h_fc1, W_fc2) + b_fc2

        return y

    def getCostFun(self, action, output_q, output_label):
        readout_action = tf.reduce_sum(tf.multiply(action, output_q), axis=1)
        cost = tf.reduce_mean(tf.square(output_label - readout_action))
        return cost

    def train(self):
        # input layer
        with tf.variable_scope('input_tensor'):
            input_tensor = tf.placeholder("float", [None, 3, 3, 1])
            output_label = tf.placeholder('float', [None])
            action = tf.placeholder('float', [None, ACTIONS])

        output_q = self.createNetwork(input_tensor)

        with tf.variable_scope('loss_function'):
            cost = self.getCostFun(action, output_q, output_label)

        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learnig_rate = 1e-8
            train_step = tf.train.GradientDescentOptimizer(
                learnig_rate).minimize(cost, global_step=global_step)

        sess = tf.Session()

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

        # 黑子取 max_q, 白子取 min_q
        while True:
            # 重新开局
            if my_game.terminal != game.TerminalStatus.GOING:
                my_game.reset()

            state = my_game.state

            # get current q value
            next_q = sess.run(output_q,
                              feed_dict={input_tensor: [state.reshape([3, 3, 1])]})[0]

            # get action index
            action_index = -1
            action_tensor = np.zeros([ACTIONS])
            if random.random() < epsilon:
                # print('-------random action----------')
                random_set = []
                for i, v in enumerate(state.reshape([-1])):
                    if v == game.IsTurnTo.BLANK.value:
                        random_set.append(i)

                action_index = random.sample(random_set, 1)[0]
            else:
                # current_state = DeepQNetwork.transferValueToOne(my_game.getState,
                #                                                 game.IsTurnTo.BLANK)
                # current_state = current_state.reshape(
                #     [game.CHESSBOARD_SIZE ** 2])
                # current_state = tf.multiply(current_state, next_q)
                # print('------------', my_game.is_turn, '--------------')
                if game.IsTurnTo.BLACK == my_game.is_turn:
                    action_index = DeepQNetwork.getMaxIndex(next_q, state.reshape([-1]),
                                                            game.IsTurnTo.BLANK.value)
                else:
                    action_index = DeepQNetwork.getMinIndex(next_q, state.reshape([-1]),
                                                            game.IsTurnTo.BLANK.value)

            action_tensor[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and times > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            print(state)
            action_position = (int(action_index / game.CHESSBOARD_SIZE),
                               int(action_index % game.CHESSBOARD_SIZE))
            print("action", action_position)
            my_game.setAction(action_position)
            next_state, reward, terminal = my_game.getState()
            side = my_game.is_turn

            if len(D) > 5000:
                D.popleft()
            D.append((state, side, action_tensor, reward, next_state, terminal))

            if times > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                state_batch = [d[0].reshape([3,3,1]) for d in minibatch]
                side_batch = [d[1] for d in minibatch]
                action_batch = [d[2] for d in minibatch]
                reward_batch = [d[3] for d in minibatch]
                next_state_batch = [d[4].reshape([3,3,1]) for d in minibatch]

                y_batch = []
                next_q_batch = sess.run(output_q,
                                        feed_dict={input_tensor: next_state_batch})

                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][5]
                    if terminal:
                        y_batch.append(reward_batch[i])
                    else:
                        next_q_value = 0
                        if side_batch[i] == game.IsTurnTo.BLACK:
                            next_q_value = DeepQNetwork.getMaxIndex(next_q_batch[i],
                                                                    next_state_batch[i].reshape([-1]),
                                                                    game.IsTurnTo.BLANK.value)
                        else:
                            next_q_value = DeepQNetwork.getMinIndex(next_q_batch[i],
                                                                    next_state_batch[i].reshape([-1]),
                                                                    game.IsTurnTo.BLANK.value)

                        y_batch.append(reward_batch[i] + GAMMA * next_q_value)

                sess.run(train_step, feed_dict={
                    output_label: y_batch,
                    input_tensor: state_batch,
                    action: action_batch
                })

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

            # if terminal != game.TerminalStatus.GOING:
            print(terminal)
            print("TIMESTEP", times, "/ STATE", state,
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward,
                "/ Q_MAX %s" % str(next_q))

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
