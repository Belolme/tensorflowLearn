import random
from math import sqrt, log as ln
from game import CHESSBOARD_SIZE, IsTurnTo
import numpy as np


def getIndexWithOp(input_t, validate_t, validate_v, op):
    """
    get index in set of input_t and validate_t equal validate_v and op is True
    """
    result = -1
    for i, validate_value in enumerate(validate_t):
        if validate_value == validate_v:
            if result == -1:
                result = i
            elif op(input_t[i], input_t[result]):
                result = i

    return result


def getMaxIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v, lambda x, y: x > y)


def getMinIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v, lambda x, y: x < y)


def boardPreprocess(board):
    result = np.zeros([CHESSBOARD_SIZE, CHESSBOARD_SIZE, 2])

    for i in range(0, CHESSBOARD_SIZE):
        for j in range(0, CHESSBOARD_SIZE):
            if board[i][j] == IsTurnTo.BLACK.value:
                result[i][j][0] = 1
            elif board[i][j] == IsTurnTo.WHITE.value:
                result[i][j][1] = 1

    return result


class UCB:
    """
    selection algorithm
    """

    def __init__(self):
        self.ex = sqrt(2)

    def __call__(self, weight, count, parent_count):
        if count == 0:
            return 1
        return weight + self.ex * sqrt(ln(parent_count) / count)


class Node:
    def __init__(self, action=None, parent=None):
        self.weight = random.random() / 20
        self.visits = 0
        self.parent = parent
        self.children = []
        self.action = action

    def lenOfChildren(self):
        return len(self.children)

    def __str__(self, level=0):
        ret = "|{}{}{}/{}\n".format("---|" * level, str(self.action),
                                    str(self.weight), str(self.visits))
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def insert(self, node):
        node.parent = self
        self.children.append(node)
