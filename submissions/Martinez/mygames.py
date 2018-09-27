from games import Game
from math import nan, isnan
from queue import PriorityQueue
from copy import deepcopy
from utils import isnumber
from grading.util import print_table
import random

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

###my Game
## 2 player 2 ship battleship game attempt
board = []

for x in range(0, 5):
    board.append(["o"] * 5)


def print_board(board):
    for row in board:
        print(" ".join(row))


player_1 = "1"
player_2 = "2"
players = [player_1, player_2]


def random_player(players):
    return random.choice(players)


def random_row(board):
    return random.randint(0, len(board) - 1)


def random_col(board):
    return random.randint(0, len(board[0]) - 1)


if random_player(players) == player_1:
    print(player_1, "starts the game.")
else:
    print(player_2, "starts the game.")

ship_row_1 = random_row(board)
ship_col_1 = random_col(board)

ship_row_2 = random_row(board)
ship_col_2 = random_col(board)

print_board(board)

player_start = random_player(players)

hit_count = 0

for turn in range(4):
    guess_row = int(input("Guess Row: 0-4) "))
    guess_col = int(input("Guess Col:0-4) "))

    if (guess_row == ship_row_1 and guess_col == ship_col_1) or (guess_row == ship_row_2 and guess_col == ship_col_2):
        hit_count = hit_count + 1
        board[guess_row][guess_col] = "*"
        print("Hit")
        if hit_count == 1:
            print("first one down")
        elif hit_count == 2:
            print("2nd and last one down . win")
            print_board(board)
            break
    else:
        if (guess_row < 0 or guess_row > 4) or (guess_col < 0 or guess_col > 4):
            print("out of bounds. try ahgain")
        elif (board[guess_row][guess_col] == "X"):
            print("You guessed that one already try to not suck")
        else:
            print("You missed ")
            board[guess_row][guess_col] = "X"
        print(turn + 1, "turn")
    print_board(board)
print("Ship 1 is hidden:")
print(ship_row_1)
print(ship_col_1)

print("Ship 2 is hidden:")
print(ship_row_2)
print(ship_col_2)

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
