import copy
import random
from game import *
from math import sqrt
from math import log as ln
from math import inf as infinity

PLAYOUT_TIMES = 10


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


ucb = UCB()


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


def selectUBCMoveNode(treeNode, gameboard):
    if gameboard.terminal == TerminalStatus.GOING:
        if gameboard.is_turn == IsTurnTo.BLACK:
            child = max(treeNode.children, key=lambda i: ucb(
                i.weight, i.visits, treeNode.visits))
        else:
            child = max(treeNode.children, key=lambda i: ucb(
                -i.weight, i.visits, treeNode.visits))
        return child
    else:
        return None


def selectionAndExpansion(treeNode, gameboard):
    game_tmp = copy.deepcopy(gameboard)
    ptr = treeNode
    while game_tmp.terminal == TerminalStatus.GOING:
        if len(ptr.children) == 0:
            validation_actions = game_tmp.getValidationAction()
            for action in validation_actions:
                ptr.insert(Node(action, ptr))

            selected_action_index = random.randint(
                0, len(validation_actions) - 1)
            ptr = ptr.children[selected_action_index]
            game_tmp.setAction(ptr.action)
            return ptr, game_tmp

        ptr = selectUBCMoveNode(ptr, game_tmp)
        game_tmp.setAction(ptr.action)

    return ptr, game_tmp


def backpropagation(tailNode, rootNode, z):
    ptr = tailNode
    while ptr is not rootNode.parent:
        ptr.weight = (ptr.weight * ptr.visits + z) / (ptr.visits + 1)
        ptr.visits += 1
        ptr = ptr.parent


def simDefault(board):
    game_tmp = copy.deepcopy(board)

    while game_tmp.terminal == TerminalStatus.GOING:
        validation_actions = game_tmp.getValidationAction()
        action_index = random.randint(0, len(validation_actions) - 1)
        action = validation_actions[action_index]
        game_tmp.setAction(action)

    _, weight, _ = game_tmp.getState()
    return weight


def mcts(treeNode, gamebaord, playout_times = PLAYOUT_TIMES):
    while treeNode.visits < playout_times:
        selected_node, game_tmp = selectionAndExpansion(treeNode, gamebaord)
        z = simDefault(game_tmp)
        backpropagation(selected_node, treeNode, z)


def getAIMoveNode(tNode, gameboard=None):
    if gameboard is not None:
        if gameboard.is_turn == IsTurnTo.BLACK:
            return max(tNode.children, key=lambda e: e.weight)
        else:
            return min(tNode.children, key=lambda e: e.weight)
    else:
        return max(tNode.children, key=lambda e: e.weight)


def playWithHuman():
    root = Node()
    my_game = TicTacToe()

    while my_game.terminal == TerminalStatus.GOING:
        mcts(root, my_game)
        root = getAIMoveNode(root)
        my_game.setAction(root.action)
        print(str(my_game), root.weight, '/', root.visits)
        # print(str(game), root.weight,'/', root.visits, 'in', [c.weight for c in root.parent.children])

        choice = input("input you position: ").split(' ')

        human_action = (int(choice[0]), int(choice[1]))
        my_game.setAction(human_action)

        for child_node in root.children:
            if child_node.action == human_action:
                root = child_node
                break

        print(str(my_game))


def MCTSTest():
    my_game = TicTacToe()
    my_game.setAction((1, 1))
    my_game.setAction((0, 1))
    my_game.setAction((0, 0))
    my_game.setAction((2, 2))
    my_game.setAction((1, 0))
    my_game.setAction((2, 0))

    # my_game.setAction((1, 2))
    # my_game.setAction((0, 2))
    # my_game.setAction((2, 1))

    print(my_game.terminal)
    print(my_game.getValidationAction())

    root = Node()
    mcts(root, my_game)
    print(root)
    print(getAIMoveNode(root, my_game).action)


if __name__ == '__main__':
    # MCTSTest()
    playWithHuman()
