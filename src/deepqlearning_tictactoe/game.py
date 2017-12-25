"""
Tic Tac Toe game
"""
from enum import Enum
import numpy as np

# win chess count
WIN_CHESS_COUNT = 3

# CHESSBOARD SIZE
CHESSBOARD_SIZE = 3


class TerminalStatus(Enum):
    """
    termianal status
    """
    BLACK_WIN = 1
    WHITE_WIN = 2
    FLAT = 3
    GOING = 4


class IsTurnTo(Enum):
    """
    is turn to enum
    """
    BLANK = 0
    WHITE = 1
    BLACK = 2

    def transfer(self):
        if self == IsTurnTo.WHITE:
            return IsTurnTo.BLACK
        elif self == IsTurnTo.BLACK:
            return IsTurnTo.WHITE
        else:
            return self


class TicTacToe:
    def __init__(self):
        self.is_turn = IsTurnTo.BLACK
        # init chessboard status
        self.state = np.zeros([CHESSBOARD_SIZE, CHESSBOARD_SIZE], dtype=int)
        self.terminal = TerminalStatus.GOING

    def reset(self):
        self.__init__()

    def setAction(self, action):
        """
        :param action: position of chessboard, data type is (x, y)
        """
        x, y = action
        if self.state[x][y] != IsTurnTo.BLANK.value:
            raise AttributeError('invalidate %d, %d' % (x, y))
        elif self.terminal != TerminalStatus.GOING:
            raise GameOverError
        else:
            self.state[x][y] = self.is_turn.value
            self.__updateTerminal(action)
            self.is_turn = IsTurnTo.WHITE if self.is_turn == IsTurnTo.BLACK else IsTurnTo.BLACK

    def getValidationAction(self):
        result = []
        if self.terminal != TerminalStatus.GOING:
            return result

        for i in range(CHESSBOARD_SIZE):
            for j in range(CHESSBOARD_SIZE):
                if self.state[i][j] == IsTurnTo.BLANK.value:
                    result.append((i, j))

        return result

    def getState(self):
        """
        :return next_state, reward, terminal(terminal status)
        """
        reward = 0
        if TerminalStatus.BLACK_WIN == self.terminal:
            reward = 1
        elif TerminalStatus.WHITE_WIN == self.terminal:
            reward = -1
        elif TerminalStatus.FLAT == self.terminal:
            reward = 0
        elif IsTurnTo.BLACK == self.is_turn:
            reward = 0
        elif IsTurnTo.WHITE == self.is_turn:
            reward = 0

        return self.state, reward, self.terminal

    def __updateTerminal(self, action):
        x, y = action
        count = 0

        def judge(nodes):
            flag = 0
            for node in nodes:
                if node == self.is_turn.value:
                    flag += 1
                else:
                    flag = 0

                if flag >= WIN_CHESS_COUNT:
                    return 1

            return 0

        # - judge
        f_p, t_p = self.__getRange(y)
        count += judge(self.state[x, f_p: t_p])

        # | judge
        f_p, t_p = self.__getRange(x)
        count += judge(self.state[f_p: t_p, y])

        # / judge
        count += judge([self.state[x + i][y + i] for i in
                        range(max(-x, -y, 1 - WIN_CHESS_COUNT), min(WIN_CHESS_COUNT, CHESSBOARD_SIZE - x, CHESSBOARD_SIZE - y))])

        # \ judge
        count += judge([self.state[x - i][y + i] for i in
                        range(max(x - CHESSBOARD_SIZE + 1, -y, 1 - WIN_CHESS_COUNT),
                              min(WIN_CHESS_COUNT, x + 1, CHESSBOARD_SIZE - y))])

        if count > 0:
            if self.is_turn == IsTurnTo.BLACK:
                self.terminal = TerminalStatus.BLACK_WIN
            else:
                self.terminal = TerminalStatus.WHITE_WIN
        elif np.min(self.state) != IsTurnTo.BLANK.value:
            self.terminal = TerminalStatus.FLAT
        else:
            self.terminal = TerminalStatus.GOING

    def __getRange(self, x):
        return 0 if x - WIN_CHESS_COUNT + 1 <= 0 else x - WIN_CHESS_COUNT + 1,\
            CHESSBOARD_SIZE if x + WIN_CHESS_COUNT >= CHESSBOARD_SIZE else x + WIN_CHESS_COUNT

    def __deepcopy__(self, memo):
        new_copy = TicTacToe()
        new_copy.is_turn = self.is_turn
        new_copy.terminal = self.terminal
        new_copy.state = self.state.copy()
        return new_copy

    def __str__(self):
        res = '    '
        for i in range(CHESSBOARD_SIZE):
            res += '%d ' % i

        res += '\n    '

        for i in range(CHESSBOARD_SIZE):
            res += '- '

        res += '\n'

        for i in range(CHESSBOARD_SIZE):
            res += '%d | ' % i
            for j in range(CHESSBOARD_SIZE):
                res += str(self.state[i][j]) + ' '
            res += '\n'

        return res


class GameOverError(Exception):
    def __str__(self):
        return 'The Game is overed.'


if __name__ == '__main__':
    game = TicTacToe()
    while game.terminal == TerminalStatus.GOING:
        print("is turn to %s terminal %s" % (game.is_turn, game.terminal))
        print(game.getState())
        next_action = tuple(int(x) for x in input().split(" "))
        game.setAction(next_action)

    print(game.terminal)
