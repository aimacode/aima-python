from collections import namedtuple
from games import (Game)
#Different board configurations can be declared here. The standard board (3 on top, 5 through the middle, 3 on bottom)
#is the "basicBoard".
basicBoard = {(0,1):[(1,2),(1,1),(1,0)],
              (1,2):[(1,1),(0,1),(2,1),(2,2)],
              (1,1):[(1,2),(0,1),(1,0),(2,1)],
              (1,0):[(1,1),(0,1),(2,1),(2,0)],
              (2,2):[(1,2),(2,1),(3,2)],
              (2,1):[(1,2),(2,2),(3,2),(3,1),(3,0),(2,0),(1,0),(1,1)],
              (2,0):[(1,0),(2,1),(3,0)],
              (3,2):[(3,1),(2,2),(2,1),(4,1)],
              (3,1):[(3,2),(3,0),(2,1),(4,1)],
              (3,0):[(2,0),(3,1),(2,1),(4,1)]
            },

class GameState:
    def __init__(self, to_move, board, bunny, wolves, label=None):
        self.to_move = to_move
        self.board = board
        self.label = label
        self.bunny = bunny
        self.wolves = wolves

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class BunnyAndWolves(Game):
    """
    Play BunnyAndWolves on an a standard board, with Max (first player) playing as the Bunny (B).
    A state has the player to move and a board, in the form of
    a dict of {(x,y): [connecting (x,y)]} entries, as well as the current locations of both the
    bunny and all wolves.
    """

    def __init__(self, h=3, v=3, k=3):
        self.initial = GameState(to_move='B', board=basicBoard, bunny=(0,1), wolves=[(3,2),(4,1),(3,0)])


    def actions(self, state):
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        if state.to_move == 'B':
            mvs = state.board[0]
            moves.append(mvs[state.bunny])
        #if state.to_move == 'W':
            #moves.append()
        state.moves = moves
        return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'B':
            return 'W'
        if player == 'W':
            return 'B'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        bb = state.board[0]
        board = bb.copy()
        player = state.to_move
        newBunny = state.bunny
        newWolves = state.wolves
        #board[move] = player
        if player == 'B':
            newBunny = move
        if player == 'W':
            newWolves = move
        next_mover = self.opponent(player)
        return GameState(to_move=next_mover, board=board, bunny=newBunny, wolves=newWolves)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        try:
            return state.utility if player == 'B' else -state.utility
        except:
            pass
        board = state.board
        #util = self.check_win(board, 'W')
        util = 0
        #if util == 0:
            #util = -self.check_win(board, 'W')
        state.utility = util
        return util if player == 'B' else -util

    # Did I win?
    def check_win(self, state, board, player):
        # check rows
        (x,y) = state.bunny
        count=0
        #for (x) in state.wolves:
        return 0

    # does player have K in a row? return 1 if so, 0 if not
    def k_in_row(self, board, start, player, direction):
        "Return true if there is a line through start on board for player."
        (delta_x, delta_y) = direction
        x, y = start
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = start
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted start itself twice
        return n >= self.k

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()


myGame = BunnyAndWolves()

won = GameState(
    to_move = 'W',
    board = basicBoard,
    bunny = (4,1),
    wolves = [(2,2),(2,1),(3,0)],
    label = 'won'
)

winin1 = GameState(
    to_move='B',
    board=basicBoard,
    bunny=(3,2),
    wolves=[(3, 1), (2, 2), (2, 1)],
    label = 'winin1'
)


losein1 = GameState(
    to_move = 'W',
    board=basicBoard,
    bunny=(2,2),
    wolves=[(1, 1), (3, 2), (2, 1)],
    label = 'losein1'
)

winin3 = GameState(
    to_move = 'W',
    board=basicBoard,
    bunny=(2,1),
    wolves=[(3, 1), (3, 2), (3, 0)],
    label = 'winin3'
)

losein3 = GameState(
    to_move='W',
    board=basicBoard,
    bunny=(1,2),
    wolves=[(2, 1), (3, 2), (1, 0)],
    label = 'losein3'
)

winin5 = GameState(
    to_move='B',
    board=basicBoard,
    bunny=(0,1),
    wolves=[(3, 2), (4, 1), (3, 0)],
    label = 'winin5'
)

lost = GameState(
    to_move='B',
    board=basicBoard,
    bunny=(2,2),
    wolves=[(1, 2), (3, 2), (2, 1)],
    label = 'lost'
)

myGames = {
    myGame: [
        won,
        winin1, losein1, winin3, losein3, winin5,
        lost,
    ]
}