from game import *


def strOfNpArray(a):
    return 'np.array(' + str(a.tolist()) + ')'


def strListChoice(choice):
    num = numChoice(choice)
    list_tmp = [0.] * 9
    list_tmp[num] = 1.
    return'np.array(' + str(list_tmp) + ')'


def strNumChoice(choice):
    return str(numChoice(choice))


def numChoice(choice):
    return choice[0] * 3 + choice[1]


if __name__ == '__main__':
    board = TicTacToe()
    output = ''

    action = [
        (1, 1), (0, 0),
        (2, 0), (0, 1),
        (2, 2), (2, 1),

        (1, 2), (0, 2)
    ]

    for a in action:
        output = output  + '(' + strOfNpArray(board.state) + ', '  \
            + 'game.' + str(board.is_turn) + ', '

        # my_choice = [int(c) for c in input('input your choose\n').split(' ')]
        # board.setAction(tuple(my_choice))
        board.setAction(a)

        _, reward, _ = board.getState()

        output = output + strListChoice(a) + ',' \
            + str(reward) + ', ' \
            + strOfNpArray(board.state) + ', ' \
            + 'game.' + str(board.terminal) + ',' \
            + strNumChoice(a) + '),\n'

    print(output)
