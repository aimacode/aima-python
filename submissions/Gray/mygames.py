from collections import namedtuple
from games import (Game)
from queue import PriorityQueue
from copy import deepcopy


class GameState:  # one way to define the state of a minimal game.
    def __init__(self, player, board, depth=8):  # add parameters as needed.
        self.label = str(board)  # change this to something easier to read
        self.board = board
        self.to_move = player
        self.Maxdepth = depth

    def __str__(self):  # use this exact signature
        return self.label


class Nim(Game):
    '''
    This is a minimal Game definition,
    the shortest implementation I could run without errors.
    '''

    def __init__(self, initial):  # add parameters if needed.
        self.initial = initial
        # add code and self.variables if needed.

    def actions(self, state):
        acts = []
        board = deepcopy(state.board)
        board_length = len(board)
        for index in range(board_length):
            for i in range(board[index]):
                acts.append((index, i + 1))
        return acts

    def result(self, state, move):
        # print('resultIn: '+ str(state.board))
        newState = deepcopy(state)
        board = newState.board
        board[move[0]] = board[move[0]] - move[1]
        newBoard = newState.board
        next_player = self.opponent(state.to_move)

        return GameState(next_player, newBoard)

    def opponent(self, to_move):
        if to_move == 'Player 1':
            return "Player 2"
        if to_move == 'Player 2':
            return "Player 1"
        return None

    def terminal_test(self, state):  # use this exact signature.
        # return True only when the state of the game is over.
        finished = False
        for i in state.board:
            if i == 0:
                finished = True

        return finished

    def utility(self, state, player):  # use this exact signature.
        try:
            return state.utility if player == 'Player 1' else -state.utility
        except:
            pass
        board = state.board
        util = self.check_win(board, 'Player 1')
        if util == 0:
            util = -self.check_win(board, 'Player 2')
        state.utility = util
        return util if player == 'Player 1' else -util

    def check_win(self, board, player):
        # check to see if the board is empty
        if board == [0,0,0]:
            return 1
        return 0

    def display(self, state):  # use this exact signature.
        board = deepcopy(state.board)
        string = ''
        board_length = len(board)
        for index in range(board_length):
            for i in range(board[index]):
                string = string + "# "
            string = string + '\n'
        print(string)


# tg = TemplateGame(TemplateState('A'))  # this is the game we play interactively.
FirstState = GameState("Player 1", [6, 7, 8])
NimGame = Nim(FirstState)

won = GameState(
    player= "Player 1",
    board= [0,0,1],

)

no_moves = GameState(
    player= "Player 1",
    board= [0,0,0],
)


myGames = {
    # myGame: [
    #     won,
    #     winin1, losein1, winin3, losein3, winin5,
    #     lost,
    # ],
    #
    # tg: [
    #     # these are the states we tabulate when we test AB(1), AB(2), etc.
    #     TemplateState('B'),
    #     TemplateState('C'),
    # ],
    NimGame: [
        FirstState,
        won,
        no_moves
    ]
}
