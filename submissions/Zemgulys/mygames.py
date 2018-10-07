from games import (Game)
from copy import deepcopy


def withinBounds(x, y):
    return x >= 0 and x <= 3 and y >= 0 and y <= 3


class GameState:
    def __init__(self, to_move, board, label=None, depth=8):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = depth

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label


class Reversi(Game):

    def __init__(self, state):
        self.initial = state

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        board = state.board
        moves = []

        for a in range(3):
            for b in range(3):
                if board[a][b] == '_':
                    moves.append([a, b])
        state.moves = moves
        return moves

    def opponent(self, player):
        if player == 'X':
            return 'O'
        if player == 'O':
            return 'X'
        return None

    def result(self, state, move):   # use this exact signature.
        newState = deepcopy(state)

        [xStart, yStart] = move
        board = newState.board
        currMover = state.to_move
        nextMover = self.opponent(currMover)

        if currMover == 'X':
            otherSide = 'O'
        else:
            otherSide = 'X'

        flipThese = []
        for xDir, yDir in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xStart, yStart
            x += xDir
            y += yDir
            if withinBounds(x, y) and board[x][y] == otherSide:

                x += xDir
                y += yDir
                if not withinBounds(x, y):
                    continue
                while board[x][y] == otherSide:
                    x += xDir
                    y += yDir
                    if not withinBounds(x, y):
                        break
                if not withinBounds(x, y):
                    continue
                while board[x][y] == currMover:
                    x -= xDir
                    y -= yDir
                    if x == xStart and y == yStart:
                        break
                    flipThese.append([x, y])

        for x, y in flipThese:
            board[x][y] = currMover

        newState.board = board
        newState.to_move = nextMover
        # use the move to modify the newState
        return newState

    def terminal_test(self, state):
        return len(self.actions(state)) == 0

    def utility(self, state, player):   # use this exact signature.
        board = state.board
        xScore = 0
        oScore = 0
        for x in range(4):
            for y in range(4):
                if board[x][y] == 'X':
                    xScore += 1
                if board[x][y] == 'O':
                    oScore += 1
        return xScore-oScore

    def display(self, state):   # use this exact signature.
        for i in range(len(state.board)):
            print(state.board[i])
        print()


reversi = GameState(
    to_move='X',
    board=[['_', '_', '_', '_'],
           ['_', 'X', 'O', '_'],
           ['_', 'O', 'X', '_'],
           ['_', '_', '_', '_']],
    label='Reversi'
)

reversi1 = GameState(
    to_move='0',
    board=[['_', '_', 'X', '_'],
           ['_', 'X', 'X', '_'],
           ['_', 'O', 'X', '_'],
           ['_', '_', '_', '_']],
    label='Reversi1'
)

reversi2 = GameState(
    to_move='X',
    board=[['_', '_', 'X', '0'],
           ['_', 'X', '0', '_'],
           ['_', 'O', 'X', '_'],
           ['_', '_', '_', '_']],
    label='Reversi2'
)

myGame = Reversi(reversi)


myGames = {
    myGame: [
        reversi, reversi1, reversi2
    ]
}











