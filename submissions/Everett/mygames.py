from collections import namedtuple
from games import (Game)

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
    def __init__(self, to_move, label, word):
        self.to_move = to_move
        self.label = label
        self.word = word

    def __str__(self):
        if self.label == None:
            return super(GhostState, self).__str__()
        return self.label

class Ghost(Game):
    """A word game that avoids the inevitable."""
    def __init__(self, initial):
        self.initial = GhostState(to_move= 'Capital', word = " ", label= 'ghost')

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

    def checkcase(self, player):
        if player.isupper():
            return 1
        if player.islower():
            return 0
        pass

    def actions(self, state):
        return upperletters

    def opponent(self, player):
        for x in upperletters:
          if player == x:
            return lowerletters
        else:
         return upperletters

    def result(self, state, move):
        if move not in self.actions(state):
          return state
        player = state.to_move
        next_mover = self.opponent(player)
        return GhostState(to_move=next_mover, label='Ghost', word=" ")

    def terminal_test(self, state):
    #    return self.utility(state, player) != 0 or len(self.actions(state)) == 0
       return None

    def utility(self, state, player):
   #     word = state.word
    #    util = self.checkifword(word, player)
     #   if util == 3:
      #      util = -self.checkifword(word, player)
       #     state.utility = util
        #return util
        return 0

    def display(self, state):
        print("Hello and welcome to a game called Ghost. Ghost is a game where the point is to not finish a word." + '\n'
        "The rules are: No made up words, Proper nouns, or Names." + '\n' 
        "Here is the wordlist for this round:" + '\n')
        printlist(Ghostwords)


def printlist(list):
    for x in list:
        bob = x
        print(bob)

eg = Ghost(GhostState('A', 'Game', 'chair'))
Ghostwords = ['chair', 'house', 'lamp']


myGames = {


    eg: [

        Ghost('B'),
    ]
}
