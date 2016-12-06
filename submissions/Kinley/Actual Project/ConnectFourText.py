from ConnectFour import ConnectFour


class ConnectFourText(ConnectFour):
    def current(self):
        print(self.players[self.first_player])

    def drop(self, column):
        ConnectFour.drop(self, column)
        print(self)
        if self.game_over != False:
            print('Game has ended.')
        return

    def __print__(self):
        s = ' ' + ''.join([str(i + 1) + ' ' for i in range(self.size['c'])]) + '\n'
        for r1 in range(self.size['r']):
            r = self.size['r'] - r1 - 1
            s += '|'
            for c in range(self.size['c']):
                if len(self.grid[c]) > r:
                    s += self.grid[c][r]
                else:
                    s += ' '
                s += '|'
            s += '\n'
        s += '+' + (2 * self.size['c'] - 1) * '-' + '+\n'
        if self.game_over == 'win':
            s += 'Player ' + self.players[self.first_player] + ' wins!'
        elif self.game_over == 'draw':
            s += 'DRAW!'
        else:
            s += 'Turn: ' + self.players[self.first_player]
        return s

    def __repr__(self):
        return self.__print__();

    def play(self):
       # print('HOW TO PLAY:\nSelect a number of column (1-' + str(
       #     self.size['c']) + ') and pass the keyboard to your opponent.\n')
        print(self)
        valid_columns = list(map(str, range(1, self.size['c'] + 1)))
        while self.game_over == False:
            c = input('Select a column: ')
            print()
            if c in valid_columns:
                self.drop(int(c) - 1)
       # print('I hope you enjoyed it!')