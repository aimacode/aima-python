from collections import namedtuple
from games import (Game)
from copy import deepcopy
from utils import isnumber
from math import nan, isnan
from queue import PriorityQueue
from utils import isnumber
from grading.util import print_table

# Make an list of words
# print at the beginning
#easy 4 letter words
#medium 5 letter words
#hard 8 letter words
#Impossible combine all the lists and have 10 letter words

#print letters each time
# Check if letter is in the alphabet
# If not input again
# each turn iterate through the word list
# if there is a word that exists in one of the turns, the last player who played loses
# if there is a set of letters that exist that doesn't lead to any other word within the list
# the last player loses.
# Start with a veryyyy small list.

# dictionary class with - check alpha, list, check if word, check if guessing
#for different difculties
#actions, a-z

lowerletters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']
upperletters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#
class GhostState:
    def __init__(self, to_move, board, Uletters, Lletters, label):
        self.to_move = to_move
        self.label = label
        self.board = []
        self.Uletters = Uletters
        self.Lletters = Lletters

    def __str__(self):
        if self.label == None:
            return super(GhostState, self).__str__()
        return self.label


class Ghost(Game):
    """A word game that avoids the inevitable."""
    def __init__(self, state):
        self.initial = state

    def actions(self, state):
        #if state.to_move == 'U':
           # moves = upperletters
         #   return moves
        #if state.to_move == 'L':
         #   moves = lowerletters
          #  return moves
       return upperletters

    def opponent(self, player):
        if player == 'U':
            return 'L'
        if player == 'U':
            return 'L'

    def result(self, state, move):
        if move not in self.actions(state):
           return state
        currMover = state.to_move
        nextMover = self.opponent(currMover)
        newState = deepcopy(state)
        newState.to_move = nextMover
        return newState


    def utility(self, state, player):
       opponent = self.opponent(player)

       return 0


    def cheating(bob, turn):
        Ghostwords = ['chair', 'house', 'lamp']
        for x in Ghostwords:
            if x[0: turn] != bob:
                return True
            else:
                return False

    def checkifword(self, player, word):
        Ghostwords = ['chair', 'house', 'lamp']
        if self.checkcase(player) == 1:
            for x in Ghostwords:
                if word.lower() == x:
                    return 3
                else:
                    return False

        if self.checkcase(player) == 0:
            for x in Ghostwords:
                if word.lower() == x:
                    return 5
                else:
                    return False
        pass

    def changeboard(thing, board):
       return board.append(thing.islower())

    def terminal_test(self, board):
     # return self.checkifword(board) == True
       return None

    def display(self, state):
        print("Hello and welcome to a game called Ghost. Ghost is a game where the point is to not finish a word." + '\n'
        "The rules are: No made up words, Proper nouns, or Names." + '\n' 
        "Here is the wordlist for this round:" + '\n')
        Ghostwords = ['chair', 'house', 'lamp']
        for x in Ghostwords:
            bob = x
            print(bob)
        print(state)

        print('Letters: ' + str(state.letter))

lose = GhostState(
    to_move = 'U',
    board = ['c','h','a','i','r'],
    label = 'Lose',
    Uletters=['c','a','r'],
    Lletters=['h','i']
)


win = GhostState(
    to_move = 'L',
    board = ['l','a','m','p'],
    label = 'Win',
    Uletters=['l', 'm'],
    Lletters=['a', 'p']
)



ghost1 = Ghost(lose)

myGames = {

    ghost1: [

        lose,
    ]
}
