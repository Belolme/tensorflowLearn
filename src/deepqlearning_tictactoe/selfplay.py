from game import *
import mcts
import tensorflow as tf
import network


def main():
    x_win = 0
    o_win = 0
    draw = 0

    input_tensor = tf.placeholder("float", [None, 3, 3, 1])
    output_label = tf.placeholder('float', [None])
    action = tf.placeholder('float', [None, CHESSBOARD_SIZE ** 2])
    q_network = network.createNetworkWithCNN1(input_tensor, True)

    sess = tf.Session()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Could not find old network weights")

    def heuristic(s, a):
        return sess.run(q_network,
                        feed_dict={
                            input_tensor: s.copy().reshape([1, 3, 3, 1])
                        })[0][a[0] * CHESSBOARD_SIZE + a[1]], 60

    for _ in range(100):
        x_tree = mcts.Node()
        o_tree = mcts.Node()
        my_game = TicTacToe()

        while my_game.terminal == TerminalStatus.GOING:
            mcts.mcts(x_tree, my_game,heuristic_fun=None, playout_times=100)
            x_tree = mcts.getAIMoveNode(x_tree, my_game)
            my_game.setAction(x_tree.action)

            for child_node in o_tree.children:
                if child_node.action == x_tree.action:
                    o_tree = child_node
                    break
            else:
                o_tree = mcts.Node()

            if my_game.terminal != TerminalStatus.GOING:
                break

            mcts.mcts(o_tree, my_game, heuristic_fun=heuristic, playout_times=100)
            o_tree = mcts.getAIMoveNode(o_tree, my_game)
            my_game.setAction(o_tree.action)

            for child_node in x_tree.children:
                if child_node.action == o_tree.action:
                    x_tree = child_node
                    break
            else:
                x_tree = mcts.Node()

        if my_game.terminal == TerminalStatus.BLACK_WIN:
            x_win += 1
        elif my_game.terminal == TerminalStatus.WHITE_WIN:
            o_win += 1
        else:
            draw += 1

        print('x win {} | o win {} | draw {}'.format(x_win, o_win, draw))

    sess.close()


if __name__ == '__main__':
    main()
