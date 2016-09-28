from collections import namedtuple
from games import (Game)

class GameState:
    def __init__(self, to_move, board, label=None):
        self.to_move = to_move
        self.board = board
        self.label = label

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class DodgeEm(Game):
    """Dodge'EM"""

    def __init__(self, h=4, v=4):
        self.h = h
        self.v = v
        self.initial = GameState(to_move='Blue', board={})
        self.openMoves = (1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4)
        self.blueWin = ((1,4), (2,4), (3,4))
        self.yellowWin =((3,4), (2,4), (1,4))

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                if (x,y) not in state.board.keys():
                    moves.append((x,y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'Blue':
            return 'Yellow'
        if player == 'Yellow':
            return 'Blue'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = state.board.copy()
        player = state.to_move
        board[move] = player
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, board=board)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        try:
            return state.utility if player == 'Blue' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, 'Blue')
        if util == 0:
            util = -self.check_win(board, 'Yellow')
        state.utility = util
        return util if player == 'Blue' else -util

    # Did I win?
    def check_win(self, board, player):
        # check Blue Win
        for (x,y) in self.blueWin[1:9]:
            if board.get((x,y)) == player:
                return 1
        # check Yellow Win
        for (x, y) in self.yellowWin[1:9]:
            if board.get((x,y)) == player:
                return 1
        return 0

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'Blue') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = DodgeEm()

blueWin = GameState(
    to_move = 'Blue',
    board = {(1,3): 'Blue', (1,1): 'Yellow', (2,3): 'Blue',
             (1,2): 'Yellow', (3,3): 'Blue', (1,3): 'Yellow',
             (4,3): 'Blue', (3,1): 'Yellow', (1,2): 'Blue', (2,1): 'Yellow', (2,2): 'Blue',
             (2,2): 'Yellow', (3,2): 'Blue', (2,3): 'Yellow', (3,3): 'Blue', (1,3): 'Yellow',
             (4,3): 'Blue', (1,4): 'Yellow', (1,3): 'Blue', (2,1): 'Yellow', (2,3): 'Blue',
             (3,1): 'Yellow', (2,3): 'Blue', (3,2): 'Yellow', (3,3): 'Blue', (2,2): 'Yellow',
             (4,3): 'Blue'
            },
    label = 'blue won'
)

yellowWin = GameState(
    to_move = 'Yellow',
    board = {(1,1): 'Yellow', (1,3): 'Blue',
             (1,2): 'Yellow', (2,3): 'Blue', (1,3): 'Yellow', (3,3): 'Blue',
             (1,4): 'Yellow', (4,3): 'Blue', (3,1): 'Yellow', (1,2): 'Blue',
             (3,2): 'Yellow', (2,2): 'Blue', (3,3): 'Yellow', (2,3): 'Blue', (3,4): 'Yellow',
             (2,4): 'Blue', (2,1): 'Yellow', (1,2): 'Blue', (2,2): 'Yellow', (1,3): 'Blue',
             (2,3): 'Yellow', (1,2): 'Blue', (2,4): 'Yellow'
            },
    label = 'yellow won'
)

myGames = {
    myGame: [
        blueWin, yellowWin
    ]
}