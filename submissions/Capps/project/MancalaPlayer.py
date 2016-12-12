import importlib
import traceback
from games import alphabeta_search
from util import roster, print_table
from io import StringIO
import sys
from tkinter import *
from PIL import Image, ImageTk
import time

root = Tk()

a = 0
b = 'No one moved yet.'
def make_ab(depth, YesOrNo):
    if YesOrNo == 'Yes':
        function = lambda game, state: alphabeta_search(state, game, d=depth, eval_fn= lambda state: game.utility4(state,'Opponent'))
    if YesOrNo == 'No':
        function = lambda game, state: alphabeta_search(state, game, d=depth, eval_fn= lambda state: game.utility4(state,'Player'))
    function.label = "AB(" + str(depth) + ")"

    return function

def printActions(actions):
    message = "Valid moves:"
    for i in range(len(actions)):
        message += ' ' + str(i + 1) + ':' + str(actions[i])
    print(message)

# def choose(number):

def PrevMov(number, player):
    if player == 'No one moved yet.':
        return player
    if number == 'Failure':
        return player + ' chose an invalid move.'
    return player + ' chose position ' + str(number + 1)

def Destroy(master):
    master.destroy()



def play_game(game, *players, verbose=False):
    """Play an n-person, move-alternating game."""



    state = game.initial
    while True:

        for player in players:
            if player == players[0]:
                while state.to_move == 'Player':
                    global b
                    global a
                    theTk = Tk()
                    label0 = Label(theTk, text=PrevMov(a, b))
                    label1 = Label(theTk, text='To move ' + player.label)
                    Button0 = Button(theTk, text=state.board.get(0), command=lambda: Press(0, theTk))
                    Button1 = Button(theTk, text=state.board.get(1), command=lambda: Press(1, theTk))
                    Button2 = Button(theTk, text=state.board.get(2), command=lambda: Press(2, theTk))
                    Button3 = Button(theTk, text=state.board.get(3), command=lambda: Press(3, theTk))
                    Button4 = Button(theTk, text=state.board.get(4), command=lambda: Press(4, theTk))
                    Button5 = Button(theTk, text=state.board.get(5), command=lambda: Press(5, theTk))
                    Button6 = Button(theTk, text=state.board.get(6), command=lambda: Press(6, theTk))
                    Button7 = Button(theTk, text=state.board.get(7), command=lambda: Press(7, theTk))
                    Button8 = Button(theTk, text=state.board.get(8), command=lambda: Press(8, theTk))
                    Button9 = Button(theTk, text=state.board.get(9), command=lambda: Press(9, theTk))
                    Button10 = Button(theTk, text=state.board.get(10), command=lambda: Press(10, theTk))
                    Button11 = Button(theTk, text=state.board.get(11), command=lambda: Press(11, theTk))
                    Button12 = Button(theTk, text=state.board.get(12), command=lambda: Press(12, theTk))
                    Button13 = Button(theTk, text=state.board.get(13), command=lambda: Press(13, theTk))

                    Button0.place(x=545, y=450)
                    Button1.place(x=595, y=450)
                    Button2.place(x=645, y=450)
                    Button3.place(x=695, y=450)
                    Button4.place(x=745, y=450)
                    Button5.place(x=795, y=450)
                    Button6.place(x=845, y=400)
                    Button7.place(x=795, y=350)
                    Button8.place(x=745, y=350)
                    Button9.place(x=695, y=350)
                    Button10.place(x=645, y=350)
                    Button11.place(x=595, y=350)
                    Button12.place(x=545, y=350)
                    Button13.place(x=495, y=400)
                    label0.place(x=495, y=300)
                    label1.place(x=495, y=500)
                    theTk.geometry('1440x900')
                    if player.label != 'Player':
                        theTk.update()
                        time.sleep(2)
                    else:
                        theTk.mainloop()
                    move = player(game, state)
                    if move != 'Failure':
                        state = game.result(state, move)
                    if verbose:
                        a = move
                        b = player.label
                        print(player.label, 'chooses', str(move))
                        game.display(state)
                    if game.terminal_test(state):
                        if not verbose:
                            game.display(state)
                        c = a
                        if c == None:
                            c = 0
                        a = game.utility(state, game.to_move(game.initial))
                        theTk = Tk()
                        label0 = Label(theTk, text=PrevMov(c, b))
                        label1 = Label(theTk, text=player.label + ' wins.')
                        label2 = Label(theTk, text='A draw.')
                        label3 = Label(theTk, text=player.label + ' lost.')
                        Button0 = Button(theTk, text=state.board.get(0), command=lambda: Destroy(theTk))
                        Button1 = Button(theTk, text=state.board.get(1), command=lambda: Destroy(theTk))
                        Button2 = Button(theTk, text=state.board.get(2), command=lambda: Destroy(theTk))
                        Button3 = Button(theTk, text=state.board.get(3), command=lambda: Destroy(theTk))
                        Button4 = Button(theTk, text=state.board.get(4), command=lambda: Destroy(theTk))
                        Button5 = Button(theTk, text=state.board.get(5), command=lambda: Destroy(theTk))
                        Button6 = Button(theTk, text=state.board.get(6), command=lambda: Destroy(theTk))
                        Button7 = Button(theTk, text=state.board.get(7), command=lambda: Destroy(theTk))
                        Button8 = Button(theTk, text=state.board.get(8), command=lambda: Destroy(theTk))
                        Button9 = Button(theTk, text=state.board.get(9), command=lambda: Destroy(theTk))
                        Button10 = Button(theTk, text=state.board.get(10), command=lambda: Destroy(theTk))
                        Button11 = Button(theTk, text=state.board.get(11), command=lambda: Destroy(theTk))
                        Button12 = Button(theTk, text=state.board.get(12), command=lambda: Destroy(theTk))
                        Button13 = Button(theTk, text=state.board.get(13), command=lambda: Destroy(theTk))

                        Button0.place(x=545, y=450)
                        Button1.place(x=595, y=450)
                        Button2.place(x=645, y=450)
                        Button3.place(x=695, y=450)
                        Button4.place(x=745, y=450)
                        Button5.place(x=795, y=450)
                        Button6.place(x=845, y=400)
                        Button7.place(x=795, y=350)
                        Button8.place(x=745, y=350)
                        Button9.place(x=695, y=350)
                        Button10.place(x=645, y=350)
                        Button11.place(x=595, y=350)
                        Button12.place(x=545, y=350)
                        Button13.place(x=495, y=400)
                        label0.place(x=495, y=300)
                        if player.label == 'Player':
                            if a == 0:
                                label2.place(x=495, y=500)
                            if a > 0:
                                label1.place(x=495, y=500)
                            if a < 0:
                                label3.place(x=495, y=500)
                        else:
                            if a == 0:
                                label2.place(x=495, y=500)
                            if a > 0:
                                label1.place(x=495, y=500)
                            if a < 0:
                                label3.place(x=495, y=500)

                        theTk.geometry('1440x900')
                        theTk.mainloop()
                        return a
                    if player.label != 'Player':
                        theTk.destroy()
            if player == players[1]:
                while (state.to_move == 'Opponent'):
                    global b
                    global a
                    theTk = Tk()
                    label0 = Label(theTk, text = PrevMov(a, b))
                    label1 = Label(theTk, text='To move ' + player.label)
                    Button0 = Button(theTk, text=state.board.get(0), command=lambda: Press(0, theTk))
                    Button1 = Button(theTk, text=state.board.get(1), command=lambda: Press(1, theTk))
                    Button2 = Button(theTk, text=state.board.get(2), command=lambda: Press(2, theTk))
                    Button3 = Button(theTk, text=state.board.get(3), command=lambda: Press(3, theTk))
                    Button4 = Button(theTk, text=state.board.get(4), command=lambda: Press(4, theTk))
                    Button5 = Button(theTk, text=state.board.get(5), command=lambda: Press(5, theTk))
                    Button6 = Button(theTk, text=state.board.get(6), command=lambda: Press(6, theTk))
                    Button7 = Button(theTk, text=state.board.get(7), command=lambda: Press(7, theTk))
                    Button8 = Button(theTk, text=state.board.get(8), command=lambda: Press(8, theTk))
                    Button9 = Button(theTk, text=state.board.get(9), command=lambda: Press(9, theTk))
                    Button10 = Button(theTk, text=state.board.get(10), command=lambda: Press(10, theTk))
                    Button11 = Button(theTk, text=state.board.get(11), command=lambda: Press(11, theTk))
                    Button12 = Button(theTk, text=state.board.get(12), command=lambda: Press(12, theTk))
                    Button13 = Button(theTk, text=state.board.get(13), command=lambda: Press(13, theTk))

                    Button0.place(x=545, y=450)
                    Button1.place(x=595, y=450)
                    Button2.place(x=645, y=450)
                    Button3.place(x=695, y=450)
                    Button4.place(x=745, y=450)
                    Button5.place(x=795, y=450)
                    Button6.place(x=845, y=400)
                    Button7.place(x=795, y=350)
                    Button8.place(x=745, y=350)
                    Button9.place(x=695, y=350)
                    Button10.place(x=645, y=350)
                    Button11.place(x=595, y=350)
                    Button12.place(x=545, y=350)
                    Button13.place(x=495, y=400)
                    label0.place(x=495, y=300)
                    label1.place(x=495, y=500)
                    theTk.geometry('1440x900')
                    if player.label != 'Player':
                        theTk.update()
                        time.sleep(2)
                    else:
                        theTk.mainloop()
                    move = player(game, state)
                    if move != 'Failure':
                        state = game.result(state, move)
                    if verbose:
                        b = player.label
                        a = move
                        print(player.label, 'chooses', str(move))
                        game.display(state)
                    if game.terminal_test(state):
                        if not verbose:
                            game.display(state)
                            #theTk.update()
                        c = a
                        if c == None:
                            c = 0
                        a = game.utility(state, game.to_move(game.initial))
                        theTk = Tk()
                        label0 = Label(theTk, text=PrevMov(c, b))
                        label1 = Label(theTk, text=player.label + ' wins.')
                        label2 = Label(theTk, text='A draw.')
                        label3 = Label(theTk, text=player.label + ' lost.')
                        Button0 = Button(theTk, text=state.board.get(0), command=lambda: theTk.destroy())
                        Button1 = Button(theTk, text=state.board.get(1), command=lambda: theTk.destroy())
                        Button2 = Button(theTk, text=state.board.get(2), command=lambda: theTk.destroy())
                        Button3 = Button(theTk, text=state.board.get(3), command=lambda: theTk.destroy())
                        Button4 = Button(theTk, text=state.board.get(4), command=lambda: theTk.destroy())
                        Button5 = Button(theTk, text=state.board.get(5), command=lambda: theTk.destroy())
                        Button6 = Button(theTk, text=state.board.get(6), command=lambda: theTk.destroy())
                        Button7 = Button(theTk, text=state.board.get(7), command=lambda: theTk.destroy())
                        Button8 = Button(theTk, text=state.board.get(8), command=lambda: theTk.destroy())
                        Button9 = Button(theTk, text=state.board.get(9), command=lambda: theTk.destroy())
                        Button10 = Button(theTk, text=state.board.get(10), command=lambda: theTk.destroy())
                        Button11 = Button(theTk, text=state.board.get(11), command=lambda: theTk.destroy())
                        Button12 = Button(theTk, text=state.board.get(12), command=lambda: theTk.destroy())
                        Button13 = Button(theTk, text=state.board.get(13), command=lambda: theTk.destroy())

                        Button0.place(x=545, y=450)
                        Button1.place(x=595, y=450)
                        Button2.place(x=645, y=450)
                        Button3.place(x=695, y=450)
                        Button4.place(x=745, y=450)
                        Button5.place(x=795, y=450)
                        Button6.place(x=845, y=400)
                        Button7.place(x=795, y=350)
                        Button8.place(x=745, y=350)
                        Button9.place(x=695, y=350)
                        Button10.place(x=645, y=350)
                        Button11.place(x=595, y=350)
                        Button12.place(x=545, y=350)
                        Button13.place(x=495, y=400)
                        label0.place(x=495, y=300)
                        if player.label == 'Player':
                            if a == 0:
                                label2.place(x=495, y=500)
                            if a < 0:
                                label1.place(x=495, y=500)
                            if a > 0:
                                label3.place(x=495, y=500)
                        else:
                            if a == 0:
                                label2.place(x=495, y=500)
                            if a < 0:
                                label1.place(x=495, y=500)
                            if a > 0:
                                label3.place(x=495, y=500)
                        theTk.geometry('1440x900')
                        theTk.mainloop()
                        return a
                    if player.label != 'Player':
                        theTk.destroy()

def query_player(game, state):
    "Make a move by querying standard input."
    actions = game.actions(state)
    printActions(actions)
    success=False
    while(not success):
        # theTk = Tk()
        # label0 = Label(theTk, text=PrevMov(a, b))
        # Button0 = Button(theTk, text=state.board.get(0), command=lambda: Press(0, theTk))
        # Button1 = Button(theTk, text=state.board.get(1), command=lambda: Press(1, theTk))
        # Button2 = Button(theTk, text=state.board.get(2), command=lambda: Press(2, theTk))
        # Button3 = Button(theTk, text=state.board.get(3), command=lambda: Press(3, theTk))
        # Button4 = Button(theTk, text=state.board.get(4), command=lambda: Press(4, theTk))
        # Button5 = Button(theTk, text=state.board.get(5), command=lambda: Press(5, theTk))
        # Button6 = Button(theTk, text=state.board.get(6), command=lambda: Press(6, theTk))
        # Button7 = Button(theTk, text=state.board.get(7), command=lambda: Press(7, theTk))
        # Button8 = Button(theTk, text=state.board.get(8), command=lambda: Press(8, theTk))
        # Button9 = Button(theTk, text=state.board.get(9), command=lambda: Press(9, theTk))
        # Button10 = Button(theTk, text=state.board.get(10), command=lambda: Press(10, theTk))
        # Button11 = Button(theTk, text=state.board.get(11), command=lambda: Press(11, theTk))
        # Button12 = Button(theTk, text=state.board.get(12), command=lambda: Press(12, theTk))
        # Button13 = Button(theTk, text=state.board.get(13), command=lambda: Press(13, theTk))
        #
        # Button0.place(x=50, y=200)
        # Button1.place(x=100, y=200)
        # Button2.place(x=150, y=200)
        # Button3.place(x=200, y=200)
        # Button4.place(x=250, y=200)
        # Button5.place(x=300, y=200)
        # Button6.place(x=350, y=150)
        # Button7.place(x=300, y=100)
        # Button8.place(x=250, y=100)
        # Button9.place(x=200, y=100)
        # Button10.place(x=150, y=100)
        # Button11.place(x=100, y=100)
        # Button12.place(x=50, y=100)
        # Button13.place(x=0, y=150)
        # label0.place(x=0, y=250)
        # theTk.geometry('450x300')
        # theTk.mainloop()
        global a
        global b
        if state.to_move == 'Player':
            if str(a) == 'quit':
                raise TiredOfPlaying()
            # try:
            index = eval(str(a))
                # assert index in range(0,len(actions)+1)
                # success=True
            # except:
            #     print('"' + str(a) + '" is not a valid index.')
            #     success = True
            if len(actions) > index >= 0:
                move = actions[index]
                return move
            else:
                return 'Failure'
        if state.to_move == 'Opponent':
            if str(a) == 'quit':
                raise TiredOfPlaying()
            # try:
            d = eval(str(a)) - 7
            index = eval(str(a)) - 7
                # assert index in range(0,len(actions)+1)
                # success=True
            # except:
            #     print('"' + str(a) + '" is not a valid index.')
            #     success = False
            if len(actions) > index >= 0:
                move = actions[index]
                return move
            else:
                return 'Failure'

query_player.label = "Player"

def Press(number, master):
    global a
    a = number
    master.destroy()


def start(YesOrNo, master, game, depth):
    ab_player = make_ab(depth, YesOrNo)
    if YesOrNo == 'Yes':
        first = query_player
        second = ab_player
    else:
        first = ab_player
        second = query_player
    master.destroy()
    score = play_game(game, first, second, verbose=True)
    print(first.label + ' wins.' if score > 0 else
          second.label + ' wins.' if score < 0 else
          'A Tie.')

def Diff(difficulty, master, game):
    try:
        depth = difficulty
        assert depth >= 0
    except:
        print('"' + difficulty + '" is not a positive integer.')
    master.destroy()
    thing = Tk()
    label0 = Label(thing, text = 'Go First?')
    label0.place(x=90, y=100)
    button0 = Button(thing, text='Yes', command= lambda:start('Yes', thing, game, depth))
    button0.place(x = 75, y=150)
    button1 = Button(thing, text = 'No', command= lambda:start('No', thing, game, depth))
    button1.place(x = 125, y = 150)
    thing.geometry('250x200')
    thing.mainloop()

def submit(master,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    master.destroy()
    submissions = {}
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + 'Johnson' + '.mygames')
        submissions['Johnson'] = mod.myGames
    except ImportError:
        pass
    except:
        traceback.print_exc()
    try:
        games = submissions['Johnson']
        arbCase(games,a,b,c,d,e,f,g,h,i,j,k,l,m,n)
    except:
        traceback.print_exc()

def arbCase(games,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    for game in games:
        game.changeGameState(a, b, c, d,
                          e, f, g, h,
                          i, j, k, l,
                          m, n)
        thing = Tk()
        label0 = Label(thing, text = 'Level of Opponent (0..8, s to skip): ')
        label0.place(x=233, y=50)
        Button0 = Button(thing, text=0, command=lambda:Diff(0, thing, game))
        Button0.place(x=100, y=100)
        Button1 = Button(thing, text=1, command=lambda:Diff(1, thing, game))
        Button1.place(x=150, y=100)
        Button2 = Button(thing, text=2, command=lambda:Diff(2, thing, game))
        Button2.place(x=200, y=100)
        Button3 = Button(thing, text=3, command=lambda:Diff(3, thing, game))
        Button3.place(x=250, y=100)
        Button4 = Button(thing, text=4, command=lambda:Diff(4, thing, game))
        Button4.place(x=300, y=100)
        Button5 = Button(thing, text=5, command=lambda:Diff(5, thing, game))
        Button5.place(x=350, y=100)
        Button6 = Button(thing, text=6, command=lambda:Diff(6, thing, game))
        Button6.place(x=400, y=100)
        Button7 = Button(thing, text=7, command=lambda:Diff(7, thing, game))
        Button7.place(x=450, y=100)
        Button8 = Button(thing, text=8, command=lambda:Diff(8, thing, game))
        Button8.place(x=500, y=100)
        thing.geometry("650x200")
        thing.mainloop()
        try_to_play(game)

label1 = Label(root, text="Position 0: Number of Stones")
label2 = Label(root, text="Position 1: Number of Stones")
label3 = Label(root, text="Position 2: Number of Stones")
label4 = Label(root, text="Position 3: Number of Stones")
label5 = Label(root, text="Position 4: Number of Stones")
label6 = Label(root, text="Position 5: Number of Stones")
label7 = Label(root, text="Position 6: Number of Stones")
label8 = Label(root, text="Position 7: Number of Stones")
label9 = Label(root, text="Position 8: Number of Stones")
label10 = Label(root, text="Position 9: Number of Stones")
label11 = Label(root, text="Position 10: Number of Stones")
label12 = Label(root, text="Position 11: Number of Stones")
label13 = Label(root, text="Position 12: Number of Stones")
label14 = Label(root, text="Position 13: Number of Stones")

E1 = Entry(root, bd=5)
E1.insert(0, 4)
E2 = Entry(root, bd=5)
E2.insert(0, 4)
E3 = Entry(root, bd=5)
E3.insert(0, 4)
E4 = Entry(root, bd=5)
E4.insert(0, 4)
E5 = Entry(root, bd=5)
E5.insert(0, 4)
E6 = Entry(root, bd=5)
E6.insert(0, 4)
E7 = Entry(root, bd=5)
E7.insert(0, 0)
E8 = Entry(root, bd=5)
E8.insert(0, 4)
E9 = Entry(root, bd=5)
E9.insert(0, 4)
E10 = Entry(root, bd=5)
E10.insert(0, 4)
E11 = Entry(root, bd=5)
E11.insert(0, 4)
E12 = Entry(root, bd=5)
E12.insert(0, 4)
E13 = Entry(root, bd=5)
E13.insert(0, 4)
E14 = Entry(root, bd=5)
E14.insert(0, 0)

label1.place(x=10, y=0)
E1.place(x=10, y=20)
label2.place(x=10, y=55)
E2.place(x=10, y=75)
label3.place(x=10, y=110)
E3.place(x=10, y=130)
label4.place(x=10, y=165)
E4.place(x=10, y=185)
label5.place(x=10, y=220)
E5.place(x=10, y=240)
label6.place(x=10, y=275)
E6.place(x=10, y=295)
label7.place(x=10, y=330)
E7.place(x=10, y=350)
label8.place(x=10, y=385)
E8.place(x=10, y=405)
label9.place(x=10, y=440)
E9.place(x=10, y=460)
label10.place(x=10, y=495)
E10.place(x=10, y=515)
label11.place(x=10, y=550)
E11.place(x=10, y=570)
label12.place(x=10, y=605)
E12.place(x=10, y=625)
label13.place(x=10, y=660)
E13.place(x=10, y=680)
label14.place(x=10, y=715)
E14.place(x=10, y=735)
root.geometry("220x800")

Submit = Button(root, text ="Submit", command = lambda:submit(root, int(E1.get()), int(E2.get()), int(E3.get()), int(E4.get()),
                          int(E5.get()), int(E6.get()), int(E7.get()), int(E8.get()),
                          int(E9.get()), int(E10.get()), int(E11.get()), int(E12.get()),
                          int(E13.get()), int(E14.get())))

Submit.place(x=10, y=770)

root.mainloop()


def try_games(games):
    for g in games:
        makeABtable(g, games[g])
        print()
        makeMoveTable(g, games[g])
        print()
        makeDisplayTable(g, games[g])
        print()
        try_to_play(g)

def try_to_play(game):
    done = False
    while not done:
        dstring = input('Level of Opponent (0..' + str(len(AB_searches)-1) + ', s to skip): ')
        if dstring.strip().lower()[0] == 's':
            break
        try:
            depth = int(dstring)
            assert depth >= 0
        except:
            print('"' + dstring + '" is not a positive integer.')
            continue
        gstring = input('Go first? [yN]: ')
        goFirst = gstring.lower().strip() == 'y'
        try:
            ab_player = make_ab(depth)
            if goFirst:
                first = query_player
                second = ab_player
            else:
                first = ab_player
                second = query_player
            score = play_game(game, first, second, verbose=True)
            print(first.label + ' wins.' if score > 0 else
                  second.label + ' wins.' if score < 0 else
                  'A Tie.')
        except TiredOfPlaying:
            pass
        except:
            traceback.print_exc()
            done = True


class TiredOfPlaying(Exception):
    pass


AB_searches = [make_ab(d) for d in range(9)]

def makeABtable(game, states):
    topLeft = str(game)[1:-1]
    header = [[str(topLeft)]]
    maxChars = len(topLeft)
    for i in range(len(AB_searches)):
        header[0].append('AB(' + str(i) + ')')
    table = []
    for state in states:
        row = [str(state)[:maxChars]]
        maxDepth = len(AB_searches)
        if hasattr(state, 'maxDepth'):
            maxDepth = min(maxDepth, state.maxDepth + 1)
        for abi in range(len(AB_searches)):
            if(abi > maxDepth):
                row.append(None)
                continue
            abSearch = AB_searches[abi]
            bestMove = abSearch(game, state)
            row.append(str(bestMove))
        table.append(row)
    print_table(table, header, tjust='rjust')

def makeMoveTable(game, states):
    header = [['state:', 'moves']]
    table = []
    for state in states:
        row = [str(state) + ':']
        moves = game.actions(state)
        moveString = str(moves)
        row.append(moveString)
        table.append(row)
    print_table(table, header)



def makeDisplayTable(game, states):
    # old_stdout = sys.stdout
    displays = []
    for state in states:
        # sys.stdout = mystdout = StringIO()
        print(state)
        game.display(state)
        # block = mystdout.getvalue()
        # displays.append(block)

    # old_stdout = sys.stdout
    for display in displays:
        lines = display.split('\n')
        print(lines)
        # mystdout.close()


class MyException(Exception):
    pass

query_player.label = "Player"

submissions = {}
scores = {}


message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.mygames')
        submissions[student] = mod.myGames
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

print(message1)
print('----------------------------------------')

for student in roster:
    if not student in submissions.keys():
        continue
    scores[student] = []
    try:
        games = submissions[student]
        print('Games from:', student)
        try_games(games)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')