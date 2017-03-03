from games import Game
from copy import deepcopy


class C4Game(Game):

    def __init__(self, state):
        self.initial = state


    def actions(self, state):
        columns = {0, 1, 2, 3, 4, 5, 6}
        # remove the columns that are full
        #.grid[column]) >= self.size['r']
        for c in [0, 1, 2, 3, 4, 5, 6]:
            if len(state.grid[c]) >= 6:
                columns.remove(c)
        return list(columns)
    #    for c in [ 0, 1, 2, 3, 4, 5, 6 ]:
    #        if state.grid[c] >= state.size['r']:
    #            columns.remove(c)
    #    return list(columns)

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


    def vertical4(self, state, player):
        for c in range(state.size['c']):
            height = len(state.grid[c])
            if height < 4:
                return 0
            count = 0
            for r in range(height):
                if state.grid[c][r] != player:
                    count += 1
                else:
                    break
            if count == 4:
                return 1
        return 0


    def horizontal4(self, state, player):
        for r in range(state.size['r']):
            width = len(state.grid[r])
            if width < 4:
                return 0
            count = 0
            for c in range(width):
                if state.grid[r][c] != player:
                    count += 1
                else:
                    break
            if count == 4:
                return 1
        return 0

    def rightDiagonal4(self, state, player):
        pass

    def leftDiagonal4(self, state, player):
        pass

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
     #   if self.vertical4(state, player):
     #       return 1
     #   if self.horizontal4(state, player):
     #       return 1
        #if self.rightDiagonal4(state, player):
        #    return 1
        #if self.leftDiagonal4(state, player):
        #    return 1


        opponent = self.opponent(player)
        if ConnectFour.drop(state,0) < 0 :
            #ConnectFour.drop(state,0)
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

     #   if self.vertical4(state, opponent):
     #       return -1
     #   if self.horizontal4(state, opponent):
     #       return -1
     #   if self.rightDiagonal4(state, opponent):
     #       return -1
     #   if self.leftDiagonal4(state, opponent):
     #       return -1

        # add other heuristics

        return 0


        # try:
        #     return state.utility if player == 'X' else -state.utility
        # except:
        #     pass
        # board = self.gameState.grid
        # util = self.check_win(board) #, 'X')
        # if util == 0:
        #     util = -self.check_win(board) #, 'O')
        # state.utility = util
        # return util if player == 'X' else -util

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        if self.utility(state, 'X') == 1:
            return True
        if self.utility(state, 'O') == 1:
            return True
        if len(self.actions(state)) == 0:
            return True
        return False




    '''
    state is an instance of ConnectFour
    move is 0..6
    '''
    def result(self, state, move):
        newState = deepcopy(state)
        # drop the move into the newState
        newState.drop(move)
        return newState

    #def utility(self, state, player):
    #    "Player relative score"
    #    return 0

    def terminal_test(self, state):
        "A state is terminal there's 4 in a row or the board is full"
        return state.check() != False


class ConnectFour:
    def __init__(self, columns=7, rows=6, player1='X', player2='O'):
        self.size = {'c': columns, 'r': rows}
        # self.grid = []
        self.first_player = True
        self.players = {True: player1, False: player2}
        self.game_over = False
        self.grid = [[] for i in range(self.size['c'])]

    def drop(self, column):
     #   if self.game_over: return False

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