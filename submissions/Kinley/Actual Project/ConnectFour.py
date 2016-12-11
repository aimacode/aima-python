from games import Game
from copy import deepcopy

class C4Game(Game):
    def __init__(self, state):
        self.initial = state

#Allows alphabeta to tell who's turn it is
    def actions(self, state):
        columns = {0, 1, 2, 3, 4, 5, 6}
        for c in [0, 1, 2, 3, 4, 5, 6]:
            if len(state.grid[c]) >= 6:
                columns.remove(c)
        return list(columns)

    # defines the order of play
    def opponent(self, player):
        if player == 'X':
            return 'O'
        if player == 'O':
            return 'X'
        return None

    def to_move(self, player):
        if player.first_player == False:
            return 'O'
        if player.first_player == True:
            return 'X'
        return None

    def utility(self, state, player):
        if ConnectFour.drop(state,0) < 0 :
            return -1
        if ConnectFour.drop(state, 1) < 0:
            return -1
        if ConnectFour.drop(state, 2) < 0:
            return -1
        if ConnectFour.drop(state, 3) < 0:
            return -1
        if ConnectFour.drop(state, 4) < 0:
            return -1
        if ConnectFour.drop(state, 5) < 0:
            return -1
        if ConnectFour.drop(state, 6) < 0:
            return -1
        if ConnectFour.drop(state, 7) < 0:
            return -1
        else:
            return 0


        # add other heuristics

        return 0


    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        if self.utility(state, 'X') == 1:
            return True
        if self.utility(state, 'O') == 1:
            return True
        if len(self.actions(state)) == 0:
            return True
        return False

    def result(self, state, move):
        newState = deepcopy(state)
        # drop the move into the newState
        newState.drop(move)
        return newState

class ConnectFour:
    def __init__(self, columns=7, rows=6, player1='X', player2='O'):
        self.size = {'c': columns, 'r': rows}
        # self.grid = []
        self.first_player = True
        self.players = {True: player1, False: player2}
        self.game_over = False
        self.grid = [[] for i in range(self.size['c'])]

    def drop(self, column):
        if column < 0 or column >= self.size['c']:
            return False
        if len(self.grid[column]) >= self.size['r']:
            return False
        self.grid[column].append(self.players[self.first_player])
        c = self.check()
        if c == False:
            self.first_player = not self.first_player
            return 1
        else:
            self.game_over = c
            return -1

    def check(self):
        d = 0
        for i, column in enumerate(self.grid):
            d += len(self.grid[i])
            for j, row in enumerate(column):
                h = i + 4 <= self.size['c']
                v = j + 4 <= len(self.grid[i])
                if v:
                    if 1 == len(set(self.grid[i][j:j + 4])):
                        return True
                if h:
                    if len(self.grid[i]) > j and len(self.grid[i + 1]) > j and len(self.grid[i + 2]) > j and len(
                            self.grid[i + 3]) > j:
                        s_r = set()
                        for k in range(4):
                            s_r.add(self.grid[i + k][j])
                        if 1 == len(s_r):
                            return True
                if h:
                    s = set()
                    for k in range(4):
                        if len(self.grid[i + k]) > j + k:
                            s.add(self.grid[i + k][j + k])
                        else:
                            s.add('??')
                    if 1 == len(s):
                        return True
                if h and j - 4 + 1 >= 0:
                    s = set()
                    for k in range(4):
                        if len(self.grid[i + k]) > j - k:
                            s.add(self.grid[i + k][j - k])
                        else:
                            s.add('??')
                    if 1 == len(s):
                        return -1
        if d == self.size['c'] * self.size['r']:
            return 'draw'
        return False