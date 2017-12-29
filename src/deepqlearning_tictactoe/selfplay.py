from game import *
import mcts
import tensorflow as tf
import network
import utils

BLACK_NETWORK_PATH = 'saved_networks_black'
WHITE_NETWORK_PATH = 'saved_networks'


def heuristic(q_network, input_tensor, sess, g, s, a, n=60):
    with sess.as_default():
        with g.as_default():
            print(sess.graph)
            print('default', tf.get_default_graph())
            return sess.run(q_network,
                    feed_dict={
                        input_tensor: [utils.boardPreprocess(s)]
                    })[0][a[0] * CHESSBOARD_SIZE + a[1]], n


def main():
    x_win = 0
    o_win = 0
    draw = 0

    black_graph = tf.Graph()
    sess_black = tf.Session(graph=black_graph)
    with sess_black.as_default():
        with black_graph.as_default():
            input_tensor_b = tf.placeholder("float", [None, 3, 3, 2])
            q_network_b = network.createNetworkWithCNN1(input_tensor_b, False)

            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(BLACK_NETWORK_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess_black, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                sess_black.run(tf.global_variables_initializer())
                print("Could not find old network weights")

            black_heuristic = lambda s, a: heuristic(q_network_b, input_tensor_b, sess_black, black_graph,
                 s, a)

    white_graph = tf.Graph()
    sess_white = tf.Session(graph=white_graph)
    with sess_white.as_default():
        with white_graph.as_default():
            input_tensor_w = tf.placeholder("float", [None, 3, 3, 2])
            q_network_w = network.createNetworkWithCNN1(input_tensor_w, False)

            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(WHITE_NETWORK_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess_white, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                sess_white.run(tf.global_variables_initializer())
                print("Could not find old network weights")

            white_heuristic = lambda s, a: heuristic(q_network_w, input_tensor_w, sess_white, white_graph,
                s, a)


    for _ in range(100):
        x_tree = mcts.Node()
        o_tree = mcts.Node()
        my_game = TicTacToe()

        while my_game.terminal == TerminalStatus.GOING:
            with sess_black.as_default():
                with black_graph.as_default():
                    # print('black', sess_black.graph)
                    mcts.mcts(x_tree, my_game, heuristic_fun=black_heuristic, playout_times=100)
            x_tree = mcts.getAIMoveNode(x_tree, my_game)
            my_game.setAction(x_tree.action)
            # print(my_game)

            for child_node in o_tree.children:
                if child_node.action == x_tree.action:
                    o_tree = child_node
                    break
            else:
                o_tree = mcts.Node()

            if my_game.terminal != TerminalStatus.GOING:
                break

            with sess_white.as_default(): 
                with white_graph.as_default():
                    # print('white', sess_white.graph)
                    mcts.mcts(o_tree, my_game, heuristic_fun=white_heuristic, playout_times=100)
            o_tree = mcts.getAIMoveNode(o_tree, my_game)
            my_game.setAction(o_tree.action)
            # print(my_game)

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

    sess_white.close()
    sess_black.close()


if __name__ == '__main__':
    main()
