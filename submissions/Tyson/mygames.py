
from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy


class myState:    # one way to define the state of a minimal game.

    def __init__(self, player, board, boardHeight, boardWidth, inARowToWin, label, depth=8): # add parameters as needed.
        self.player = player
        self.to_move = player
        self.board = board
        self.boardHeight = boardHeight
        self.boardWidth = boardWidth
        self.inARowToWin = inARowToWin
        self.label = label # change this to something easier to read
        self.maxDepth = depth
        # add code and self.variables as needed.

    def __str__(self):  # use this exact signature
        return self.label

# class TemplateAction:
#     '''
#     It is not necessary to define an action.
#     Start with actions as simple as a label (e.g., 'Down')
#     or a pair of coordinates (e.g., (1,2)).
#
#     Don't un-comment this until you already have a working game,
#     and want to play smarter.
#     '''
#     def __lt__(self, other):    # use this exact signature
#         # return True when self is a better move than other.
#         return False

class Connect4(Game):
    '''
    This is a minimal Game definition,
    the shortest implementation I could run without errors.
    '''

    def __init__(self, initial):    # add parameters if needed.
        self.initial = initial


        # add code and self.variables if needed.

    def actions(self, state):   # use this exact signature.
        acts = []
        rowsState = []
        rowsAlreadyUsed = []
        for y in state.board.keys():
            rowsState.append(y)
        for a in rowsState:
            rowsAlreadyUsed.append(a[0])

        for d in range(1, state.boardWidth):  #trying to get rowsState to have all empty and used rows
            if rowsAlreadyUsed.__contains__(d):
                continue
            else:
                emptyColumn = (d, 0)
                rowsState.append(emptyColumn)


        for z in rowsState:  #appends all legal actions
            if z[1] < state.boardHeight:
                 validMove = (z[0], z[1]+1)
                 acts.append(validMove)

                    # validMove = (x,1)
                    # acts.append(validMove)





        # append all moves, which are legal in this state,
        # to the list of acts.
        return acts

    def opponent(self, player):
        if player == 'R':
            return 'B'
        if player == 'B':
            return 'R'
        return None

    def result(self, state, move):   # use this exact signature.

        newState = deepcopy(state)
        newState.board.update({ move: newState.player})
        # use the move to modify the newState
        newState.player = self.opponent(newState.player)
        newState.to_move = newState.player
        # newState.board = self.board

        #newState.board = newState.board.append(move)

        return newState






    def utility(self, state, player):   # use this exact signature.
        ''' return:
        >0 if the player is winning,
        <0 if the player is losing,
         0 if the state is a tie.
        '''
        try:
            return state.utility if player == 'R' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_for_win(board, 'R', state)
        if util == 0:
            util = -self.check_for_win(board, 'B', state)
        state.utility = util
        return util if player == 'R' else -util



    def check_for_win(self, board, player, state): #only checks for horizontal and vertical wins right now
        # check rows
        for y in range(1, state.boardHeight + 1):
            if self.k_in_row(board, (1,y), player, (1,0), state):
                return 1
        # check columns
        for x in range(1, state.boardWidth + 1):
            if self.k_in_row(board, (x,1), player, (0,1), state):
                return 1

        return 0



    def k_in_row(self, board, start, player, direction, state):
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
        return n >= state.inARowToWin

    def terminal_test(self, state):   # use this exact signature.
        # return True only when the state of the game is over.
        return self.utility(state, 'R') != 0 or len(self.actions(state)) == 0



    def display(self, state):   # use this exact signature.
        # pretty-print the game state, using ASCII art,
        # to help a human player understand his options.
        board = state.board
        for y in range(state.boardHeight, 0, -1):
            for x in range(1, state.boardWidth+1):
                if (x, y) in board:
                    playerToPrint = board[x,y] #board.get(x, y)
                    print(playerToPrint, end=' ')

                else:
                    if(y<1):
                        continue
                    else:
                        print('.', end=' ')
            print()
        print()





won = myState(player = 'B',
              board = {
                  # (1,2): 'Red',
                  (1,2): 'B',
                  (1,1): 'R', (2,1): 'R',
              },
              boardHeight = 2,
              boardWidth = 3,
              inARowToWin = 2,
              label= 'won'

              )  # where the game is already won
winIn1 = myState(
              player = 'R',
              board = {
                  (1,2): 'B', (2,2): 'B',
                  (1,1): 'R', (2,1): 'R',
              },
              boardHeight = 4,
              boardWidth = 5,
              inARowToWin= 3,
              label= 'winIn1'

  ) # one move from a win

play = myState(player = 'R',
              board = {},
              boardHeight = 6,
              boardWidth = 7,
              inARowToWin= 4,
              label= 'Normal Game'

              ) # the start state of a normal game


# lost = myState(...) # where the game is already lost
# tied = myState(...) # the game has ended in a tie
# won1 = myState(...) # one move from a win

playableGame = Connect4(play)

myGames = {
    playableGame: [
         won,
         winIn1,
         play
    ]
}