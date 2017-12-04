"""
Tic Tac Toe game
"""
from enum import Enum
import numpy as np


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


# win chess count
WIN_CHESS_COUNT = 3

# CHESSBOARD SIZE
CHESSBOARD_SIZE = 3


class TicTacToe:
    def __init__(self):
        self.is_turn = IsTurnTo.BLACK
        self.state = np.zeros([3, 3], dtype=int)  # init chessboard status
        self.terminal = TerminalStatus.GOING

    def setAction(self, action):
        """
        :param action: position of chessboard, data type is (x, y)
        """
        x, y = action
        if self.state[x][y] != IsTurnTo.BLANK.value:
            raise AttributeError
        elif self.terminal != TerminalStatus.GOING:
            raise GameOverError
        else:
            self.state[x][y] = self.is_turn.value
            self.__updateTerminal(action)
            self.is_turn = IsTurnTo.WHITE if self.is_turn == IsTurnTo.BLACK else IsTurnTo.BLACK

    def getState(self, chess_type):
        """
        :return next_state, reward, terminal(terminal status)
        """
        reward = 0
        if chess_type == IsTurnTo.BLACK and self.terminal == TerminalStatus.BLACK_WIN \
                or chess_type == IsTurnTo.WHITE and self.terminal == TerminalStatus.WHITE_WIN:
            reward = 1
        elif chess_type == IsTurnTo.BLACK and self.terminal == TerminalStatus.WHITE_WIN \
                or chess_type == IsTurnTo.WHITE and self.terminal == TerminalStatus.BLACK_WIN:
            reward = -1

        return self.state, reward, self.terminal

    def __updateTerminal(self, action):
        if np.min(self.state) != IsTurnTo.BLANK.value:
            self.terminal = TerminalStatus.FLAT
            return

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
            else:
                return 0

        # - judge
        f_p, t_p = self.__getRange(y)
        count += judge(self.state[x, f_p: t_p])

        # | judge
        f_p, t_p = self.__getRange(x)
        count += judge(self.state[f_p: t_p, y])

        # / judge
        count += judge([self.state[x + i][y + i] for i in
                        range(max(-x, -y, -2), min(3, CHESSBOARD_SIZE - x, CHESSBOARD_SIZE - y))])

        # \ judge
        count += judge([self.state[x - i][y + i] for i in
                        range(max(x - CHESSBOARD_SIZE + 1, -y, -2),
                              min(3, x + 1, CHESSBOARD_SIZE - y))])

        if count > 0:
            if self.is_turn == IsTurnTo.BLACK:
                self.terminal = TerminalStatus.BLACK_WIN
            else:
                self.terminal = TerminalStatus.WHITE_WIN

    def __getRange(self, x):
        return 0 if x - WIN_CHESS_COUNT + 1 <= 0 else x - WIN_CHESS_COUNT + 1,\
            CHESSBOARD_SIZE if x + WIN_CHESS_COUNT >= CHESSBOARD_SIZE else x + WIN_CHESS_COUNT


class GameOverError(Exception):
    def __str__(self):
        return 'The Game is overed.'


if __name__ == '__main__':
    game = TicTacToe()
    while game.terminal == TerminalStatus.GOING:
        print("is turn to %s terminal %s" % (game.is_turn, game.terminal))
        print(game.state)
        next_action = tuple(int(x) for x in input().split(" "))
        game.setAction(next_action)

    print(game.terminal)
