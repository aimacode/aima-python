from collections import namedtuple
from games import (Game)

class GameState:
    def __init__(self, to_move, board, label=None, depth=8):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.maxDepth = depth
        self.score = {'CM': 0}

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class GameOfKings(Game):
    """A very simplified game of Chess, where you and your opponent
        play as kings. Just like in Chess, you are able to move your
        king one space in any direction and you win by taking out the
        opponent's king."""

    def __init__(self, h=4, v=4):
        self.h = h
        self.v = v
        self.initial = GameState(to_move='E', board={(1,1): 'E', (4,4): 'F'})
        self.englandWin = (1,1)
        self.franceWin = (4,4)

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square within one space in any direction."
        moves = []
        for x in range(1, self.h - 1):
            for y in range(1, self.v - 1):
                if (x, y) not in state.board.keys():
                    moves.append((x, y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'E':
            return 'F'
        if player == 'E':
            return 'F'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = player
        player = state.to_move
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, board=board)

    def utility(self, state, player, move):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        if player == 'E' and state.score['CM'] == 100:
            if self.opponent(player) == next_mover:
                return 1
            else:
                return -1
        elif player == 'F' and state.score['CM'] == 100:
            if self.opponent(player) == next_mover:
                return 1
            else:
                return -1
        return 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        if state.score['CM'] == 100:
            return 1
        return 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = GameOfKings()

won = GameState(
    to_move = 'F',
    board = {(2,2): 'E', (2,2): 'F'
            },
    label = 'won'
)

winin1 = GameState(
    to_move = 'E',
    board = {(2,2): 'E', (3,3): 'F'
            },
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'F',
    board = {(2,2): 'E', (3,3): 'F'
            },
    label = 'losein1'
)

losein3 = GameState(
    to_move = 'F',
    board = {(1,2): 'E', (4,4): 'F'
            },
    label = 'losein3'
)

winin3 = GameState(
    to_move = 'E',
    board = {(1,2): 'E', (4,4): 'F'
            },
    label = 'winin3'
)

lost = GameState(
    to_move = 'E',
    board = {(2,2): 'E', (2,2): 'F'
            },
    label = 'lost'
)

myGames = {
    myGame: [
        won,
        winin1, losein1, winin3, losein3,
        lost,
    ]
}