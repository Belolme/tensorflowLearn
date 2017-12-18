import copy
import random
from game import *
from math import sqrt
from math import log as ln
from math import inf as infinity


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


def monteCarloTreeSearch(treeNode, gameboard):
    ucb = UCB()

    while treeNode.visits < 100000:
        ptr = treeNode
        game_tmp = copy.deepcopy(gameboard)

        # Selection
        while len(ptr.children) > 0:
            if game_tmp.is_turn == IsTurnTo.BLACK:
                index, child = max(enumerate(ptr.children), key=lambda i: ucb(
                    i[1].weight, i[1].visits, ptr.visits))
            else:
                index, child = max(enumerate(ptr.children), key=lambda i: ucb(
                    -i[1].weight, i[1].visits, ptr.visits))
            ptr = ptr.children[index]
            game_tmp.setAction(child.action)

        # Expansion
        minmax = False

        if game_tmp.terminal == TerminalStatus.GOING:
            validationAction = game_tmp.getValidationAction()
            for a in validationAction:
                ptr.insert(Node(a))

            randSelect = random.randint(0, len(validationAction) - 1)
            ptr = ptr.children[randSelect]

            # Simulation
            action = ptr.action
            while True:
                game_tmp.setAction(action)
                if game_tmp.terminal != TerminalStatus.GOING:
                    break
                action = random.sample(
                    game_tmp.getValidationAction(), 1)[0]

            winner = game_tmp.terminal
        else:
            # 走到这里的时候说明整一棵树已经构建完成了，使用 minmax 进行反向传播
            minmax = True

        # Backpropagation
        if minmax:
            ptr.visits += 1
            is_turn = game_tmp.is_turn
            
            _, new_weight, _ = game_tmp.getState()
            ptr.weight = new_weight

            while not (ptr is treeNode):
                if is_turn == IsTurnTo.BLACK:
                    new_weight_child = min(ptr.parent.children,
                                     key=lambda i: i.weight)
                else:
                    new_weight_child = max(ptr.parent.children,
                                     key=lambda i: i.weight)

                ptr = ptr.parent
                ptr.visits += 1
                ptr.weight = new_weight_child.weight
                is_turn = is_turn.transfer()
        else:
            _, new_weight, _ = game_tmp.getState()
            while not (ptr is treeNode.parent):
                ptr.weight = (ptr.weight * ptr.visits +
                              new_weight) / (ptr.visits + 1)
                ptr.visits += 1
                ptr = ptr.parent


def getAIMoveNode(tNode, game=None):
    if game is not None:
        if game.terminal == IsTurnTo.BLACK:
            return max(tNode.children, key=lambda e: e.weight)
        else:
            return min(tNode.children, key=lambda e: e.weight)
    else:
        return max(tNode.children, key=lambda e: e.weight)


def playWithMCTS():
    root = Node()
    game = TicTacToe()

    while game.terminal == TerminalStatus.GOING:
        monteCarloTreeSearch(root, game)
        root = getAIMoveNode(root)
        game.setAction(root.action)
        print(str(game), root.weight, '/', root.visits)
        # print(str(game), root.weight,'/', root.visits, 'in', [c.weight for c in root.parent.children])

        choice = input("input you position: ").split(' ')

        humanAction = (int(choice[0]), int(choice[1]))
        game.setAction(humanAction)

        for childNode in root.children:
            if childNode.action == humanAction:
                root = childNode
                break

        print(str(game))


def MCTSTest():
    game = TicTacToe()
    game.setAction((1, 1))
    game.setAction((0, 1))
    game.setAction((0, 0))
    game.setAction((2, 2))
    game.setAction((1, 0))
    game.setAction((0, 2))
    
    # game.setAction((1, 2))
    # game.setAction((2, 0))
    # game.setAction((2, 1))

    print(game.terminal)
    print(game.getValidationAction())

    root = Node()
    monteCarloTreeSearch(root, game)
    print(root)
    print(getAIMoveNode(root, game).action)


if __name__ == '__main__':
    # MCTSTest()
    playWithMCTS()
  