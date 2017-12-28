import tensorflow as tf
from game import TicTacToe
from game import IsTurnTo
from game import TerminalStatus
from game import CHESSBOARD_SIZE
import mcts
import network
import utils

VALIDATE_TIMES = 100
MCTS_SIMULATION_TIMES = 100


def validate(sess=None):
    input_tensor = tf.placeholder("float", [None, 3, 3, 2])
    output_label = tf.placeholder('float', [None])
    action = tf.placeholder('float', [None, CHESSBOARD_SIZE ** 2])

    q_network = network.createNetworkWithCNN1(input_tensor, True)
    train_step, loss = network.getTrainStepAndLossFun(
        action, q_network, output_label)
    summ = tf.summary.merge_all()

    if sess is None:
        sess = tf.Session()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Could not find old network weights")

    mcts_win = 0
    mcts_lost = 0
    draw = 0

    for i in range(VALIDATE_TIMES):
        my_game = TicTacToe()
        tree = mcts.Node()
        print('############## %d times ####################' % i)

        while my_game.terminal == TerminalStatus.GOING:
            mcts.mcts(tree, my_game, playout_times=MCTS_SIMULATION_TIMES)
            tree = mcts.getAIMoveNode(tree, my_game)
            my_game.setAction(tree.action)

            # print(str(my_game))

            if my_game.terminal != TerminalStatus.GOING:
                break

            q_value = sess.run(q_network, feed_dict={
                input_tensor: [utils.boardPreprocess(my_game.state)]
            })[0]

            if my_game.is_turn == IsTurnTo.BLACK:
                index = utils.getMaxIndex(q_value, my_game.state.copy().reshape([9]),
                                          IsTurnTo.BLANK.value)
            else:
                index = utils.getMinIndex(q_value, my_game.state.copy().reshape([9]),
                                          IsTurnTo.BLANK.value)

            network_action = (int(index / CHESSBOARD_SIZE),
                              int(index % CHESSBOARD_SIZE))
            my_game.setAction(network_action)

            for child_node in tree.children:
                if child_node.action == network_action:
                    tree = child_node
                    break

            # print(str(my_game))

        print(my_game.terminal)
        if my_game.terminal == TerminalStatus.BLACK_WIN:
            mcts_win += 1
        elif my_game.terminal == TerminalStatus.WHITE_WIN:
            mcts_lost += 1
        else:
            draw += 1

    sess.close()
    return mcts_win, mcts_lost, draw


if __name__ == '__main__':
    print(validate())
