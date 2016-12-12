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


    def __init__(self, h=14):
        self.h = h
        self.initial = GameState(to_move='Player', board={(0): 4, (1): 4, (2): 4, (3): 4, (4): 4, (5): 4, (6): 0,
                                                          (7): 4, (8): 4, (9): 4, (10): 4, (11): 4, (12): 4, (13): 0})

    def changeGameState(self,a = 4, b = 4, c = 4, d = 4, e = 4, f = 4, g = 0, h = 4, i = 4, j = 4, k = 4,
                        l = 4, m = 4, n = 0, o = 'Player'):
        self.initial = GameState(to_move= o, board={(0): a, (1): b, (2): c, (3): d, (4): e, (5): f, (6): g,
                                                    (7): h, (8): i, (9): j, (10): k, (11): l, (12): m, 13: n})


    def actions(self, state, id=0):
        if(id == 1):
            moves = []
            player = state.to_move
            try:
                return state.moves
            except:
                pass

            if player == 'Player':
                for x in range(0, int(self.h/2 - 1)):
                    if state.board.get(x) != 0:
                        moves.append(x)

            if player == 'Opponent':
                for x in range(int(self.h/2), int(self.h - 1)):
                    if state.board.get(x) != 0:
                        moves.append(x)


            state.moves = moves
            return moves
        if (id == 0):
            moves = []
            player = state.to_move
            try:
                return state.moves
            except:
                pass

            if player == 'Player':
                for x in range(0, int(self.h / 2 - 1)):
                    #if state.board.get(x) != 0:
                    moves.append(x)

            if player == 'Opponent':
                for x in range(int(self.h / 2), int(self.h - 1)):
                   # if state.board.get(x) != 0:
                    moves.append(x)

            state.moves = moves
            return moves

    # defines the order of play
    def opponent(self, player):
        if player == 'Player':
            return 'Opponent'
        if player == 'Opponent':
            return 'Player'
        return None

    def result(self, state, move):
        if move not in self.actions(state):
            return state  # Illegal move has no effect
        board = state.board.copy()
        player = state.to_move

        if player == 'Player':
            for x in range(0, int(self.h/2 - 1)):
                if (x) == move:
                    y = board.get(x)
                    if y==0:
                        return state  # Illegal move has no effect

                    #takes all pebbles out of hole and put one in each hole counterclockwise (excluding opponent's
                    #mancala).
                    board[x] = 0
                    while y != 0:
                        if (x + 1) % self.h != self.h - 1:
                            board[((x + 1) % self.h)] += 1
                            y -= 1
                        x += 1

                    #if the last pebble lands in the player's mancala, player gets another turn.
                    if x % self.h == int(self.h/2 - 1):
                        next_mover = player
                        return GameState(to_move=next_mover, board=board)

                    #if the last pebble lands in a hole with no pebble already in it,
                    #and the hole adjacent to that hole has pebbles in it,
                    #take all the pebbles.
                    if int(x % self.h) < int(self.h/2):
                        if board[((x) % self.h)] == 1:
                            a = (self.h - 2) - ((x) % self.h)
                            if board[a] != 0:
                                b = board[a]
                                board[a] = 0
                                board[int(self.h / 2 - 1)] = board[int(self.h / 2 - 1)] + b + board[((x) % self.h)]
                                board[((x) % self.h)] = 0
            next_mover = self.opponent(player)
            return GameState(to_move=next_mover, board=board)





        if player == 'Opponent':
            for x in range(int(self.h/2), self.h - 1):
                if (x) == move:
                    y = board.get(x)
                    if y==0:
                        return state  # Illegal move has no effect

                    # takes all pebbles out of hole and put one in each hole counterclockwise (excluding opponent's
                    # mancala).
                    board[x] = 0
                    while y != 0:
                        if (x + 1) % self.h != int(self.h / 2 - 1):
                            board[((x + 1) % self.h)] += 1
                            y -= 1
                        x += 1

                    # if the last pebble lands in the player's mancala, player gets another turn.
                    if x % self.h == int(self.h - 1):
                        next_mover = player
                        return GameState(to_move=next_mover, board=board)

                    # if the last pebble lands in a hole with no pebble already in it,
                    # and the hole adjacent to that hole has pebbles in it,
                    # take all the pebbles.
                    if int(x % self.h) >= int(self.h / 2):
                        if board[((x) % self.h)] == 1:
                            a = (self.h - 2) - ((x) % self.h)
                            if board[a] != 0:
                                b = board[a]
                                board[a] = 0
                                board[int(self.h - 1)] = board[int(self.h - 1)] + b + board[((x) % self.h)]
                                board[((x) % self.h)] = 0
            next_mover = self.opponent(player)
            return GameState(to_move=next_mover, board=board)




    def utility(self, state, player):
        try:
            return state.utility if player == 'Player' else -state.utility
        except:
            pass
        board = state.board
        util = 0
        if self.terminal_test(state):
            #put the remaining stones into the corresponding player's mancala
            for x in range(0, int(self.h/2-1)):
                if board.get(x) != 0:
                    board[int(self.h/2-1)] += board.get(x)
                    board[x] = 0

            for x in range(int(self.h/2), int(self.h - 1)):
                if board.get(x) != 0:
                    board[int(self.h - 1)] += board.get(x)
                    board[x] = 0

        # a is the winning condition, first heuristic
        a = board.get(self.h - 1) - board.get(int(self.h/2 - 1))
        if player == 'Player':
            util = -a
        else:
            util = a


        # second heuristic. b is number of stones on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that keep pieces on one side of the board
        # and prioritizes moves that prevent the opponent from making capture moves.

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     b += board.get(x)
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     c += board.get(x)
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        # third heuristic. b is number of empty on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that will result in more moves, and prioritizes moves
        # that prevent the opponent from making capture moves

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     if board.get(x) == 0:
        #         b += 1
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     if board.get(x) == 0:
        #         c += 1
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        # fourth heuristic. b is number of empty on player's side, c is number on opponent's

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     y = board.get(x) % self.h
        #     d = (self.h - 2) - ((x) % self.h)
        #     if y < int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
        #         b += board[d]
        #
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     y = board.get(x) % self.h
        #     d = (self.h - 2) - ((x) % self.h)
        #     if y > int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
        #         c += board[d]
        #
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        return util

    def utility1(self, state, player):
        try:
            return state.utility if player == 'Player' else -state.utility
        except:
            pass
        board = state.board
        util = 0
        if self.terminal_test(state):
            #put the remaining stones into the corresponding player's mancala
            for x in range(0, int(self.h/2-1)):
                if board.get(x) != 0:
                    board[int(self.h/2-1)] += board.get(x)
                    board[x] = 0

            for x in range(int(self.h/2), int(self.h - 1)):
                if board.get(x) != 0:
                    board[int(self.h - 1)] += board.get(x)
                    board[x] = 0

        # a is the winning condition, first heuristic
        a = board.get(self.h - 1) - board.get(int(self.h/2 - 1))
        if player == 'Player':
            util = -a
        else:
            util = a


        # second heuristic. b is number of stones on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that keep pieces on one side of the board
        # and prioritizes moves that prevent the opponent from making capture moves.

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     b += board.get(x)
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     c += board.get(x)
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        # third heuristic. b is number of empty on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that will result in more moves, and prioritizes moves
        # that prevent the opponent from making capture moves

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     if board.get(x) == 0:
        #         b += 1
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     if board.get(x) == 0:
        #         c += 1
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        # fourth heuristic. b is number of empty on player's side, c is number on opponent's

        # b = 0
        # c = 0
        # for x in range(0, int(self.h/2 - 1)):
        #     y = board.get(x) % self.h
        #     d = (self.h - 2) - ((x) % self.h)
        #     if y < int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
        #         b += board[d]
        #
        # for x in range(int(self.h/2), int(self.h - 1)):
        #     y = board.get(x) % self.h
        #     d = (self.h - 2) - ((x) % self.h)
        #     if y > int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
        #         c += board[d]
        #
        # if player == 'Player':
        #     util = -a + (b - c)
        # else:
        #     util = a + (c - b)

        return util

    def utility2(self, state, player):
        try:
            return state.utility if player == 'Player' else -state.utility
        except:
            pass
        board = state.board
        util = 0
        if self.terminal_test(state):
            #put the remaining stones into the corresponding player's mancala
            for x in range(0, int(self.h/2-1)):
                if board.get(x) != 0:
                    board[int(self.h/2-1)] += board.get(x)
                    board[x] = 0

            for x in range(int(self.h/2), int(self.h - 1)):
                if board.get(x) != 0:
                    board[int(self.h - 1)] += board.get(x)
                    board[x] = 0

        # a is the winning condition, first heuristic
        a = board.get(self.h - 1) - board.get(int(self.h/2 - 1))


        # second heuristic. b is number of stones on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that keep pieces on one side of the board
        # and prioritizes moves that prevent the opponent from making capture moves.

        b = 0
        c = 0
        for x in range(0, int(self.h/2 - 1)):
            b += board.get(x)
        for x in range(int(self.h/2), int(self.h - 1)):
            c += board.get(x)
        if player == 'Player':
            util = -a + (b - c)
        else:
            util = a + (c - b)

        return util

    def utility3(self, state, player):
        try:
            return state.utility if player == 'Player' else -state.utility
        except:
            pass
        board = state.board
        util = 0
        if self.terminal_test(state):
            #put the remaining stones into the corresponding player's mancala
            for x in range(0, int(self.h/2-1)):
                if board.get(x) != 0:
                    board[int(self.h/2-1)] += board.get(x)
                    board[x] = 0

            for x in range(int(self.h/2), int(self.h - 1)):
                if board.get(x) != 0:
                    board[int(self.h - 1)] += board.get(x)
                    board[x] = 0

        # a is the winning condition, first heuristic
        a = board.get(self.h - 1) - board.get(int(self.h/2 - 1))

        # third heuristic. b is number of empty on player's side, c is number on opponent's
        # this heuristic seems to prioritize moves that will result in more moves, and prioritizes moves
        # that prevent the opponent from making capture moves

        b = 0
        c = 0
        for x in range(0, int(self.h/2 - 1)):
            if board.get(x) == 0:
                b += 1
        for x in range(int(self.h/2), int(self.h - 1)):
            if board.get(x) == 0:
                c += 1
        if player == 'Player':
            util = -a + (b - c)
        else:
            util = a + (c - b)

        return util

    def utility4(self, state, player):
        try:
            return state.utility if player == 'Player' else -state.utility
        except:
            pass
        board = state.board
        util = 0
        if self.terminal_test(state):
            #put the remaining stones into the corresponding player's mancala
            for x in range(0, int(self.h/2-1)):
                if board.get(x) != 0:
                    board[int(self.h/2-1)] += board.get(x)
                    board[x] = 0

            for x in range(int(self.h/2), int(self.h - 1)):
                if board.get(x) != 0:
                    board[int(self.h - 1)] += board.get(x)
                    board[x] = 0

        # a is the winning condition, first heuristic
        a = board.get(self.h - 1) - board.get(int(self.h/2 - 1))

        # fourth heuristic. b is number of empty on player's side, c is number on opponent's

        b = 0
        c = 0
        for x in range(0, int(self.h/2 - 1)):
            y = board.get(x) % self.h
            d = (self.h - 2) - ((x) % self.h)
            if y < int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
                b += board[d]

        for x in range(int(self.h/2), int(self.h - 1)):
            y = board.get(x) % self.h
            d = (self.h - 2) - ((x) % self.h)
            if y > int(self.h / 2) and board.get(y) == 0 and board.get(d) != 0:
                c += board[d]

        if player == 'Player':
            util = -a + (b - c)
        else:
            util = a + (c - b)

        return util


    # Did I win?

    def check_win(self, board, player):
        n = 0
        if player == 'Player':
            for x in range(0, int(self.h/2) - 1):
                if board.get((x)) != 0:
                    n += 1
            return n == 0
        for x in range(int(self.h / 2), int(self.h) - 1):
            if board.get((x)) != 0:
                n += 1
        return n == 0

    # does player have all their pieces off the board? Return true if more than one piece on board.
    # def board_piece_check(self, board, start, player):
    #     n=0
    #     for x in range (1, self.v + 1):
    #         for y in range (1, self.h + 1):
    #             if board == player:
    #                 n += 1
    #     return n == 0

    def terminal_test(self, state):
        return self.check_win(state.board, 'Player') or self.check_win(state.board, 'Opponent')

    def display(self, state):
        board = state.board
        #str = '  '
        #spaces = ''
        #for x in range(int(self.h/2 - 1), (self.h - 1)):
            #str = str + board.get(x) + ' '
            #spaces = ''
        #print(str)
        #print
        print('  ',board.get(12),' ' , board.get(11), ' ' ,  board.get(10), ' ' ,  board.get(9) , ' ' ,  board.get(8) ,
                ' ' , board.get(7))
        print(board.get(13), '                       ', board.get(6))
        print('  ' , board.get(0) , ' ' , board.get(1) , ' ' , board.get(2) , ' ' , board.get(3) ,
              ' ' , board.get(4), ' ', board.get(5))




myGame = FlagrantCopy()

winin1 = GameState(
    to_move = 'Opponent',
    board = {(0): 0, (1): 0, (2): 0, (3): 0, (4): 0, (5):0,
             (6): 4, (7): 1, (8): 1, (9): 1, (10):1,(11):1, (12): 1, (13): 1
            },
    label = 'winin1'
)

myGames = {
    myGame: [
        #won,
         winin1,
        #losein1, winin3, losein3, winin5,
        # lost,
    ]
}


#
myGame.display(winin1)
#print(myGame.terminal_test(winin1))
#print(myGame.actions(winin1))
# print('\n')
#winin1 = myGame.result(winin1,(5))
myGame.display(winin1)
print(myGame.actions(winin1))
#winin1 = myGame.result(winin1,(7))
#myGame.display(winin1)
#print(myGame.actions(winin1))
#winin1 = myGame.result(winin1,(4))
#myGame.display(winin1)
#print(myGame.actions(winin1))
