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


class FlagrantCopy(Game):
    """Game of Hex.
    The goal of the game is for one player to cut
    the other off from making a complete connection to their side.
    X has the vertical, or connecting the top and bottom.
    O has the horizontal, or connecting the left and right."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        self.initial = GameState(to_move='X', board={})

    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        for x in range(0, self.h):
            for y in range(0, self.v):
                if (x, y) not in state.board.keys():
                    moves.append((x, y))
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'X':
            return 'O'
        if player == 'O':
            return 'X'
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
            return state.utility if player == 'X' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, 'X')
        if util == 0:
            util = -self.check_win(board, 'O')
        state.utility = util
        return util if player == 'X' else -util

    # Did I win?
    def check_win(self, board, player):
        # check vertical line
        if player == 'X':
            return self.check_connected(board, player)
        # check horizontal
        if player == 'O':
            return self.check_connected(board, player)
        return 0

    def check_connected(self, board, player):
        if player == 'X':
            for a in range(0, self.h):
                coor = (0, a)

                try:  # if the coordinate does not exist
                    if board[coor] == 'X':
                        tree = [coor]
                        surr = self.get_surrounding(coor, tree, board, player)
                        won = self.recur(surr, tree, board, player)
                        if won == 1:
                            return 1
                        else:
                            continue
                except:
                    pass
            return 0

        if player == 'O':
            for a in range(0, self.v):
                coor = (a, 0)

                try:  # if the coordinate does not exist
                    if board[coor] == 'O':
                        tree = [coor]
                        surr = self.get_surrounding(coor, tree, board, player)
                        won = self.recur(surr, tree, board, player)
                        if won == 1:
                            return 1
                        else:
                            continue
                except:
                    pass
            return 0

    def recur(self, surrounding, tree, board, player):
        # Reached end of tree, but did not reach target
        if len(surrounding) < 1:
            return 0

        elif not self.hit_target(surrounding, player):
            for s in surrounding:
                tree.append(s)
                surr = self.get_surrounding(s, tree, board, player)
                won = self.recur(surr, tree, board, player)
                if won == 1:
                    return 1
                else:
                    continue
        else:
            return 1
        return 0

    def hit_target(self, surrounding, player):
        for cor in surrounding:
            # Check vertical
            if player == 'X':
                r, _ = cor
                if r == self.v - 1:
                    return True
            # check horizontal
            else:  # if player is O
                _, c = cor
                if c == self.h - 1:
                    return True
        return False

    def get_surrounding(self, coor, tree, board, player):
        surrounding = []
        y, x = coor
        for row in range(y - 1, y + 2):
            for col in range(x - 1, x + 2):
                # don't count self
                if (row, col) in tree:
                    continue

                try:  # coordinate is out of bounds
                    if board[(row, col)] == player:
                        surrounding.append((row, col))
                except:
                    pass

        return surrounding

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.h):
            for y in range(0, self.v):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = FlagrantCopy(4, 4, 4)

won = GameState(
    to_move='O',
    board={
        (0, 2): 'X',
        (1, 0): 'O', (1, 1): 'O', (1, 2): 'X',
        (2, 1): 'O', (2, 2): 'X',
        (3, 2): 'X',
    },
    label='won'
)
win1 = GameState(
    to_move='X',
    board={
        (0, 2): 'X',
        (1, 0): 'O', (1, 1): 'O', (1, 2): 'X',
        (2, 1): 'O', (2, 2): 'X',
    },
    label='win1'
)
win2 = GameState(
    to_move='X',
    board={
        (1, 0): 'O', (1, 2): 'X',
        (2, 0): 'O', (2, 1): 'O', (2, 2): 'X',

    },
    label='win2'
)
win3 = GameState(
    to_move='X',
    board={
        (0, 0): 'O', (1, 2): 'X',
    },
    label='win3'
)

lose = GameState(
    to_move='X',
    board={
        (0, 2): 'X',
        (1, 0): 'O', (1, 1): 'O', (1, 2): 'O',
        (2, 1): 'X', (2, 2): 'X', (2, 3): 'O',
    },
    label='lose'
)
lose1 = GameState(
    to_move='O',
    board={
        (0, 2): 'X',
        (1, 0): 'O', (1, 1): 'O', (1, 2): 'O',
        (2, 1): 'X', (2, 2): 'X',
    },
    label='lose1'
)

myGames = {
    myGame: [
        won,
        win1,
        win2,
    ]
}
