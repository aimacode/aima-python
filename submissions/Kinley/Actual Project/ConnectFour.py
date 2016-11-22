class ConnectFour:
    def __init__(self, columns=7, rows=6, player1='X', player2='O'):
        self.size = {'c': columns, 'r': rows}
        self.grid = []
        self.first_player = True
        self.players = {True: player1, False: player2}
        self.game_over = False
        self.grid = [[] for i in range(self.size['c'])]

    def drop(self, column):
        if self.game_over: return False

        if column < 0 or column >= self.size['c']:
            return False
        if len(self.grid[column]) >= self.size['r']:
            return False

        self.grid[column].append(self.players[self.first_player])

        c = self.check()
        if c == False:
            self.first_player = not self.first_player
            return True
        else:
            self.game_over = c
            return True

    def check(self):
        d = 0
        for i, column in enumerate(self.grid):
            d += len(self.grid[i])
            for j, row in enumerate(column):
                h = i + 4 <= self.size['c']
                v = j + 4 <= len(self.grid[i])

                if v:
                    if 1 == len(set(self.grid[i][j:j + 4])):
                        return 'win'

                if h:
                    if len(self.grid[i]) > j and len(self.grid[i + 1]) > j and len(self.grid[i + 2]) > j and len(
                            self.grid[i + 3]) > j:
                        s_r = set()
                        for k in range(4):
                            s_r.add(self.grid[i + k][j])
                        if 1 == len(s_r):
                            return 'win'

                if h:
                    s = set()
                    for k in range(4):
                        if len(self.grid[i + k]) > j + k:
                            s.add(self.grid[i + k][j + k])
                        else:
                            s.add('??')
                    if 1 == len(s):
                        return 'win'

                if h and j - 4 + 1 >= 0:
                    s = set()
                    for k in range(4):
                        if len(self.grid[i + k]) > j - k:
                            s.add(self.grid[i + k][j - k])
                        else:
                            s.add('??')
                    if 1 == len(s):
                        return 'win'

        if d == self.size['c'] * self.size['r']:
            return 'draw'

        return False