from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table

class GameState:
    def __init__(self, to_move, position, board, label=None):
        self.to_move = to_move
        self.position = position
        self.board = board
        self.label = label
        self.scores = {'H':0, 'V': 0}

    def __str__(self):
        if self.label == None:
            return super(GameState, self).__str__()
        return self.label

class Move:
    def __init__(self, r, c, v):
        self.row = r
        self.col = c
        self.value = v

    def rcv(self):
        return self.row, self.col, self.value

    def __lt__(self, other):
        return self.value > other.value

def q2list(mq):
    list = []
    while not mq.empty():
        list.append(mq.get(1).rcv())
    return list

def movesInRow(board, r):
    mQueue = PriorityQueue()
    row = board[r]
    for c in range(len(row)):
        if isnan(row[c]):
            continue
        v = row[c]
        move = Move(r,c,v)
        mQueue.put(move)
    return q2list(mQueue)

def movesInCol(board, c):
    mQueue = PriorityQueue()
    for r in range(len(board)):
        if isnan(board[r][c]):
            continue
        v = board[r][c]
        move = Move(r,c,v)
        mQueue.put(move)
    return q2list(mQueue)

class ThinkAhead(Game):
    """
    An implementation of ThinkAhead
    """
    def __init__(self, state):
        self.initial = state

    def actions(self, state):
        "Legal moves are any square not yet taken."
        r, c = state.position
        if state.to_move == 'H':
            moves = movesInRow(state.board, r)
            return moves
        if state.to_move == 'V':
            moves = movesInCol(state.board, c)
            return moves
        return []

    # defines the order of play
    def opponent(self, player):
        if player == 'H':
            return 'V'
        if player == 'V':
            return 'H'
        return None

    def result(self, state, move):
        r, c, v = move
        assert state.board[r][c] == v
        currMover = state.to_move
        nextMover = self.opponent(currMover)

        newState = deepcopy(state)
        newState.to_move = nextMover
        newState.position = r, c
        newState.board[r][c] = nan
        newState.scores[currMover] += v
        return newState

    def utility(self, state, player):
        "Player relative score"
        opponent = self.opponent(player)
        return state.scores[player] - state.scores[opponent]

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return len(self.actions(state)) == 0

    def display(self, state):
        print_table(state.board, njust='center', sep=',')

        print('Score: ' + str(state.scores))

won = GameState(
    to_move = 'H',
    position = (0,1),
    board = [[nan,nan],
             [ 9 ,nan]],
    label = 'won'
)
won.scores = {'H':9, 'V': 0}

lost = GameState(
    to_move = 'V',
    position = (0,1),
    board = [[nan,nan],
             [ 9 ,nan]],
    label = 'lost'
)
lost.scores = {'H':0, 'V': 9}

winin1 = GameState(
    to_move = 'H',
    position = (1,1),
    board = [[nan,nan],
             [ 9 ,nan]],
    label = 'winin1'
)

losein1 = GameState(
    to_move = 'V',
    position = (0,0),
    board = [[nan,nan],
             [ 9 ,nan]],
    label = 'losein1'
)

winin2 = GameState(
    to_move = 'H',
    position = (0,0),
    board = [[nan, 3 , 2 ],
             [nan, 9 ,nan],
             [nan,nan, 1]],
    label = 'winin2'
)

losein2 = GameState(
    to_move = 'V',
    position = (0,0),
    board = [[nan,nan,nan],
             [ 3 , 9 ,nan],
             [ 2 ,nan, 1]],
    label = 'losein2'
)
losein2.maxDepth = 3

# http://www.kongregate.com/games/zolli/thinkahead-brain-trainer
stolen = GameState(
    to_move = 'H',
    position = (3,1),
    board = [[3, 8 ,9,5],
             [9, 1 ,3,2],
             [8, 6 ,4,4],
             [9,nan,1,5]],
    label = 'stolen'
)

choose1 = GameState(
    to_move = 'H',
    position = (1,0),
    board = [[ 3 , 8 ,9,5],
             [nan, 1 ,3,2],
             [ 8 , 6 ,4,4],
             [nan,nan,1,5]],
    label = 'choose1'
)

winby10 = GameState(
    to_move='H',
    position=(2, 0),
    board=[[nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan,  6 ,  4 ,  5 ],
           [nan, nan,  1 ,  3 ]],
    label = 'winby10'
)

thinkA = ThinkAhead(stolen)

myGames = {
    thinkA: [
        won,
        lost,
        winin1,
        losein1,
        winin2,
        losein2,
        stolen,
        choose1,
        winby10
    ]
}
