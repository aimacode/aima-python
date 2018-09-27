from collections import namedtuple
from games import (Game)
from copy import deepcopy

class TemplateState:    # one way to define the state of a minimal game.

    def __init__(self, player, board):  # add parameters as needed.
        self.to_move = player
        self.label = str(id(self))   # change this to something easier to read
        self.board = board
        # add code and self.variables as needed.
        # print(board)

    def __str__(self):  # use this exact signature
        return self.label

class TemplateGame(Game):
    '''
    This is a minimal Game definition,
    the shortest implementation I could run without errors.
    '''

    def __init__(self, initial):    # add parameters if needed.
        self.initial = initial
        # add code and self.variables if needed.

        self.initial = TemplateState('1', [[4, 4, 4, 4, 4, 4, 0], [4, 4, 4, 4, 4, 4, 0]])

    def actions(self, state):   # use this exact signature.
        try:
            return state.moves
        except:
            pass
        "Legal moves are any square not yet taken."
        moves = []
        board = state.board
        for x in range(0,6):
            if board[0][x] > 0:
                moves.append(x+1)
            if board[1][x] > 0:
                moves.append(x + 7)
        return moves



    def result(self, state, move):   # use this exact signature.
        newState = deepcopy(state)
        # if move not in self.actions(state):
        #     return state  # Illegal move has no effect
        board = state.board.copy()
        firstMove = move
        player = newState.to_move
        lastMove = 0
        if player == '1' and move < 7:
            count1 = board[0][move-1]
            game = 7 - move
            board[0][move-1] = 0
            while count1 > 0 and game != 0:
                board[0][firstMove] = board[0][firstMove] + 1
                count1 -= 1
                game -= 1
                lastMove = firstMove
                firstMove += 1


            while count1 > 0:
                for y in range(0, 6):
                    board[1][y] = board[1][y] + 1
                    count1 -= 1
                    lastMove = y
                if count1 > 0:
                    for z in range(0,7):
                        board[0][z] = board[0][z] + 1
                        count1 -= 1
                        lastMove = z
        if player == '2' and move > 6:
            count1 = board[1][move - 7]
            game = 6 - (move - 7)
            board[1][move - 7] = 0
            while count1 > 0 and game != 0:
                board[1][firstMove - 7] = board[1][firstMove - 7] + 1
                count1 -= 1
                game -= 1
                lastMove = firstMove - 7
                firstMove += 1

            while count1 > 0:
                for y in range(0, 7):
                    board[0][y] = board[0][y] + 1
                    count1 -= 1
                    lastMove = y
                if count1 > 0:
                    for z in range(0,6):
                        board[1][z] = board[1][z] + 1
                        count1 -= 1
                        lastMove = z
        if lastMove == 7:
            newState.to_move = newState.to_move
        else:
          #  state.to_move = self.opponent(player)
            newState.to_move = self.opponent(player)
        newState.board = board
        return newState

    # defines the order of play
    def opponent(self, player):
        if player == '1':
            return '2'
        if player == '2':
            return '1'
        return None

    def terminal_test(self, state):   # use this exact signature.
        board = state.board
        player = self.opponent(state.to_move)
        countOne = 0
        countTwo = 0
        for x in range(0, 6):
            countOne += board[0][x]
        for x in range(0, 6):
            countTwo += board[1][x]
        if player == '1' and countOne == 0 and board[0][6] > board[1][6]:
            return True
        if player == '1' and countTwo == 0 and board[0][6] > board[1][6]:
            return True
        if player == '2' and countOne == 0 and board[1][6] > board[0][6]:
            return True
        if player == '2' and countTwo == 0 and board[1][6] > board[0][6]:
            return True
        return False

    def utility(self, state, player):   # use this exact signature.
        board = state.board
        player = self.opponent(state.to_move)
        countOne = 0
        countTwo = 0
        for x in range(0, 6):
            countOne += board[0][x]
        for x in range(0, 6):
            countTwo += board[1][x]
        if player == '1' and countOne == 0 and board[0][6] > board[1][6]:
            return 1
        if player == '1' and countTwo == 0 and board[0][6] > board[1][6]:
            return 1
        if player == '2' and countOne == 0 and board[1][6] > board[0][6]:
            return 2
        if player == '2' and countTwo == 0 and board[1][6] > board[0][6]:
            return 2
        return False
        return 0

    def display(self, state):   # use this exact signature.
        # pretty-print the game state, using ASCII art,
        # to help a human player understand his options.
        board = state.board
        print(' - - - - - - - -')
        print('  |'+str(board[1][5])+'|'+str(board[1][4])+'|'+str(board[1][3])+'|'+str(board[1][2])+'|'+str(board[1][1])+'|'+str(board[1][0])+'|')
        print('|' + str(board[1][6]) +'|'+'           '+'|'+ str(board[0][6]) +'|')
        print('  |' + str(board[0][0]) + '|' + str(board[0][1]) + '|' + str(board[0][2]) + '|' + str(board[0][3]) + '|' + str(board[0][4]) + '|' + str(board[0][5]) + '|')
        print(' - - - - - - - -')

#tg = TemplateGame(TemplateState('1', [[4, 4, 4, 4, 4, 4, 0], [4, 4, 4, 4, 4, 4, 0]]))   # this is the game we play interactively.



won = TemplateState(
    '1',
    [[0,0,0,0,0,0,30],[0,0,3,0,0,0,15]]
)
winin1 = TemplateState(
    '1',
    [[0, 0, 0, 0, 0, 1, 29], [0, 0, 3, 1, 1, 0, 13]]
)

losein1 = TemplateState(
    '2',
    [[0, 0, 0, 3, 0, 0, 14], [0, 0, 0, 0, 0, 1, 30]]
)

winin3 = TemplateState(
    '2',
    [[0, 0, 0, 0, 2, 0, 30],[0, 0, 0, 2, 0, 0, 14]]
)


real = TemplateState(
    '1',
    [[4, 4, 4, 4, 4, 4, 0], [4, 4, 4, 4, 4, 4, 0]]
)


#
# losein3 = TemplateState(
#     to_move = 'O',
#     board = {(1,1): 'X',
#              (2,1): 'X',
#              (3,1): 'O', (1,2): 'X', (1,2): 'O',
#             },
#     label = 'losein3'
# )
#
# winin5 = TemplateState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'O',
#              (2,1): 'X',
#             },
#     label = 'winin5'
# )
#
# lost = TemplateState(
#     to_move = 'X',
#     board = {(1,1): 'X', (1,2): 'X',
#              (2,1): 'O', (2,2): 'O', (2,3): 'O',
#              (3,1): 'X'
#             },
#     label = 'lost'
# )

myGame = TemplateGame(won)

myGames = {
    myGame: [
        #won,
        #winin1,
        #losein1,
        #winin3, #losein3, winin5,
        real
        #lost,
    ],

    # tg: [
    #     # these are the states we tabulate when we test AB(1), AB(2), etc.
    #    # TemplateState('B',[[4, 4, 4, 4, 4, 4, 0], [4, 4, 4, 4, 4, 4, 0]]),
    #     #TemplateState('C',[[4, 4, 4, 4, 4, 4, 0], [4, 4, 4, 4, 4, 4, 0]]),
    # ]
}
