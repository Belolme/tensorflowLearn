"""
tic tac toe game ai
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
import game
import network
import utils
import mcts

ACTIONS = 9  # number of valid actions
GAMMA = 1  # decay rate of past observations
OBSERVE = 1024  # timesteps to observe before training
EXPLORE = 3000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
REPLAY_MEMORY = 8000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
TRAIN_STEP = 10
NETWORK_PATH = 'saved_networks_black'


class DeepQNetwork:
    def getOutputQLabel(self, batch, sess, y, input_tensor):
        turn_to_batch = [d[1] for d in batch]
        reward_batch = [d[3] for d in batch]
        next_state_batch = [d[4] for d in batch]

        label_batch = []
        for i in range(0, len(batch)):
            terminal = batch[i][5]

            if terminal != game.TerminalStatus.GOING:
                label_batch.append(reward_batch[i])
            else:
                next_q_t = sess.run(y, feed_dict={input_tensor: [
                                    utils.boardPreprocess(next_state_batch[i])]})[0]
                if turn_to_batch[i] == game.IsTurnTo.BLACK:
                    next_q_value = next_q_t[utils.getMinIndex(next_q_t,
                                                              next_state_batch[i].copy(
                                                              ).reshape([-1]),
                                                              game.IsTurnTo.BLANK.value)]
                    # next_q_value = -1 if next_q_value < -1 else next_q_value
                else:
                    next_q_value = next_q_t[utils.getMaxIndex(next_q_t,
                                                              next_state_batch[i].copy(
                                                              ).reshape([-1]),
                                                              game.IsTurnTo.BLANK.value)]
                    # next_q_value = 1 if next_q_value > 1 else next_q_value

                # print(next_q_t)
                # print(next_q_value)
                label_batch.append(next_q_value)

        return label_batch

    def train(self):
        # input layer
        with tf.variable_scope('input_tensor'):
            input_tensor = tf.placeholder("float", [None, 3, 3, 2])
            output_label = tf.placeholder('float', [None])
            action = tf.placeholder('float', [None, ACTIONS])

        output_q = network.createNetworkWithCNN1(input_tensor, True)
        train_step, loss = network.getTrainStepAndLossFun(
            action, output_q, output_label)
        summ = tf.summary.merge_all()

        win_statistic = tf.placeholder('float', [3])
        black_win_rate = win_statistic[0] / win_statistic[2]
        white_win_rate = win_statistic[1] / win_statistic[2]
        draw_rate = (win_statistic[2] - win_statistic[0] -
                     win_statistic[1]) / win_statistic[2]

        win_statistic_summ = tf.summary.merge([tf.summary.scalar('black_win_rate', black_win_rate),
                                               tf.summary.scalar('white_win_rate', white_win_rate),
                                               tf.summary.scalar('draw', draw_rate)])

        win_rate_statis_queue = deque()

        with tf.Session() as sess:
            # saving and loading networks
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(NETWORK_PATH)
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

            writer = tf.summary.FileWriter(
                './datasets/log', tf.get_default_graph())

            # 黑子取 max_q, 白子取 min_q
            while True:
                # 重新开局
                if my_game.terminal != game.TerminalStatus.GOING:
                    win_rate_statis_queue.append(my_game.terminal)
                    if len(win_rate_statis_queue) > 10:
                        win_rate_statis_queue.popleft()

                    print(draw_rate, black_win_rate, white_win_rate)
                    black_win_c = 0
                    white_win_c = 0

                    for t in win_rate_statis_queue:
                        if t is game.TerminalStatus.BLACK_WIN:
                            black_win_c += 1
                        elif t is game.TerminalStatus.WHITE_WIN:
                            white_win_c += 1

                    deno = len(win_rate_statis_queue)

                    _, _, _, summ_output = sess.run([black_win_rate, white_win_rate, draw_rate, win_statistic_summ], feed_dict={
                             win_statistic: [black_win_c, white_win_c, deno]})
                    writer.add_summary(summ_output, times)
                    # print("black win rate: %f | white win rate %f | draw rate: %f"  \
                    # % tuple(sess.run([black_win_rate, white_win_rate, draw_rate], feed_dict={win_statistic: [black_win_c, white_win_c, deno]})))

                    my_game.reset()

                state = my_game.state.copy()
                is_turn_to = my_game.is_turn

                # get current q value
                next_q = sess.run(output_q,
                                  feed_dict={input_tensor: [utils.boardPreprocess(state)]})[0]

                # get action index
                action_index = -1
                action_tensor = np.zeros([ACTIONS])
                # if random.random() < epsilon:
                if is_turn_to == game.IsTurnTo.WHITE or random.random() < epsilon:
                    print('-------random action----------')
                    # random_set = []
                    # for i, v in enumerate(state.reshape([-1])):
                    #     if v == game.IsTurnTo.BLANK.value:
                    #         random_set.append(i)

                    # action_index = random.sample(random_set, 1)[0]
                    tree = mcts.Node()
                    mcts.mcts(tree, my_game, playout_times=100)
                    mcts_action = mcts.getAIMoveNode(tree, my_game).action
                    action_index = mcts_action[0] * 3 + mcts_action[1]
                else:
                    print('------------q value action--------------')
                    if game.IsTurnTo.BLACK == is_turn_to:
                        action_index = utils.getMaxIndex(next_q, state.reshape([-1]),
                                                         game.IsTurnTo.BLANK.value)
                    else:
                        action_index = utils.getMinIndex(next_q, state.reshape([-1]),
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

                new_m = (state.copy(), is_turn_to, action_tensor.copy(),
                         reward, next_state.copy(), terminal)
                if len(D) < REPLAY_MEMORY:
                    for d in D:
                        if np.array_equal(new_m[0], d[0]) and np.array_equal(new_m[2], d[2]):
                            break
                    else:
                        D.append(new_m)
                print('D lenght is: ', len(D))
                print('new_m is ', new_m)

                if len(D) > OBSERVE:
                    # sample a minibatch to train on
                    minibatch = random.sample(D, int(BATCH))

                    # while True:
                    output_q_batch = self.getOutputQLabel(
                        minibatch, sess, output_q, input_tensor)
                    # print(output_q_b_batch)
                    _, loss_output, summ_output = sess.run([train_step, loss, summ], feed_dict={action: [d[2] for d in minibatch],
                                                                                                input_tensor: [utils.boardPreprocess(d[0]) for d in minibatch],
                                                                                                output_label: output_q_batch})

                    writer.add_summary(summ_output, times)

                    if loss_output < 0.05 and epsilon <= FINAL_EPSILON:
                        epsilon = INITIAL_EPSILON

                    times += 1

                    # save progress every 10000 iterations
                    if times % 10000 == 0:
                        saver.save(sess, NETWORK_PATH + "/tictactoe-dqn",
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


if __name__ == "__main__":
    DeepQNetwork().train()
