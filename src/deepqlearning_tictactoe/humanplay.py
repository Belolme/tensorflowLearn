import os
import mcts
import network
import utils
import game
import tensorflow as tf

BLACK_NETWORK_PATH = 'saved_networks_black'
WHITE_NETWORK_PATH = 'saved_networks'


def playWithRawNetwork(human_side, network_path):
    os.system('cls')
    board = game.TicTacToe()

    input_tensor = tf.placeholder("float", [None, 3, 3, 2])
    q_network = network.createNetworkWithCNN1(input_tensor, False)

    sess = tf.Session()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(network_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Could not find old network weights")

    turn_to_human = False
    if human_side == game.IsTurnTo.BLACK:
        turn_to_human = True

    while board.terminal == game.TerminalStatus.GOING:
        if turn_to_human:
            choice = tuple([int(c)
                            for c in input('put your action\n').split(' ')])
            board.setAction(choice)
            turn_to_human = not turn_to_human
            print(board, end='')

        if board.terminal != game.TerminalStatus.GOING:
            break

        q_value = sess.run(q_network,
                           feed_dict={input_tensor: [utils.boardPreprocess(board.state)]})[0]
        if board.is_turn == game.IsTurnTo.BLACK:
            index = utils.getMaxIndex(q_value, board.state.copy().reshape([9]),
                                      game.IsTurnTo.BLANK.value)
        else:
            index = utils.getMinIndex(q_value, board.state.copy().reshape([9]),
                                      game.IsTurnTo.BLANK.value)
        board.setAction((int(index / game.CHESSBOARD_SIZE),
                         int(index % game.CHESSBOARD_SIZE)))
        os.system('cls')
        print(board, end='')

        turn_to_human = not turn_to_human

    print(board.terminal)


if __name__ == '__main__':
    playWithRawNetwork(game.IsTurnTo.WHITE, WHITE_NETWORK_PATH)
