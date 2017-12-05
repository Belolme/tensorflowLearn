"""
tic tac toe game ai
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
import game.tictactoegame as game

ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10.  # timesteps to observe before training
EXPLORE = 200.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
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
        return tf.nn.DeepQNetwork.__conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

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
            W_fc1 = DeepQNetwork.__weightVariable([64, 128])
            b_fc1 = DeepQNetwork.__biasVariable([128])
            h_conv2_flat = tf.reshape(h_conv2, [-1, 64])
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

    def train(self, black=True):
        # input layer
        with tf.variable_scope('input_tensor'):
            input_tensor = tf.placeholder("int", [BATCH, 3, 3, 1])
            output_label = tf.placeholder('float', [None])
            action = tf.placeholder('int', [None, ACTIONS])

        output_q = self.createNetwork(input_tensor)

        with tf.variable_scope('loss function'):
            cost = self.getCostFun(action, output_q, output_label)

        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learnig_rate = 1e-4
            train_step = tf.train.GradientDescentOptimizer(
                learnig_rate).minimize(cost, global_step=global_step)

        D = deque()

        sess = tf.Session()

        # saving and loading networks
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks_%s" % 'black' if black else 'white')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())
            print("Could not find old network weights")

        # init game
        my_game = game.TicTacToe()
        
        if black:
            game_state = my_game.getState()
        else:
            random_action = random.randint(0, 8)

            
        
        # start training
        epsilon = INITIAL_EPSILON
        times = 0

        while True:


