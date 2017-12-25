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

    while board.terminal == TerminalStatus.GOING:
        output = output + ', ' + '(' + strOfNpArray(board.state) + ', '  \
            + 'game.' + str(board.is_turn) + ', '

        my_choice = [int(c) for c in input('input your choose\n').split(' ')]
        board.setAction(tuple(my_choice))

        _, reward, _ = board.getState()

        output = output + strListChoice(my_choice) + ',' \
            + str(reward) + ', ' \
            + strOfNpArray(board.state) + ', ' \
            + 'game.' + str(board.terminal) + ',' \
            + strNumChoice(my_choice) + ')'

    print(output)


