"""
Tic Tac Toe game
"""
import numpy as np

# termianal status
BLACK_WIN = 1
WHITE_WIN = 2
FLAT = 3
GOING = 4

# is turn enum
BLANK = 0
WHITE = 1
BLACK = 2

# win chess count
WIN_CHESS_COUNT = 3

# CHESSBOARD SIZE
CHESSBOARD_SIZE = 3


class TicTacToe:

    def init(self):
        self.isTurn = BLACK
        self.state = np.zeros([3, 3])  # init chessboard status
        self.terminal = GOING

    def setAction(self, action):
        """
        :param action: position of chessboard, data type is (x, y)
        """
        x, y = action
        if self.state[x][y] != BLANK:
            raise AttributeError
        elif self.terminal != GOING:
            raise GameOverError
        else:
            self.state[x][y] = isTurn
            self.isTurn = WHITE if self.isTurn == BLACK else BLACK
            self.__updateTerminal(action)

    def getNextState(self):
        """
        :return next_state, reword, terminal(terminal status)
        """

        return self.state,

    def __updateTerminal(self, action):
        if np.min(self.state) != BLANK:
            self.terminal = FLAT
            return

        x, y = action
        count = 0

        def judge(nodes):
            flag = 0
            for node:
                nodes
                if node == self.isTurn:
                    flag += 1
                else:
                    flag = 0
            if flag >= WIN_CHESS_COUNT:
                self.terminal = self.isTurn
                return 1
            else:
                return 0

        # - judge
        count += judge(self.state[x， __getRange(y)])

        # | judge
        count += judge(self.state[__getRange(x), y])

        # / judge
        i = __getRange()
        count += judge(self.state[][])

    def __getRange(x):
        return 0 if x - WIN_CHESS_COUNT - 1 < 0 else x - WIN_CHESS_COUNT - 1,\
            CHESSBOARD_SIZE if x + WIN_CHESS_COUNT >= CHESSBOARD_SIZE else x + WIN_CHESS_COUNT


class GameOverError(Exception):

    def __str__(self):
        return 'The Game is overed.'
