from tkinter import *
import ConnectFour
from ConnectFour import C4Game
from random import randint
import games

g = C4Game

# class GameState:
#     def __init__(self, to_move, board, label=None, depth=8):
#         self.to_move= to_move
#         self.board = board
#         self.label = label
#         self.maxDepth = depth
# 
#         def __str__(self):
#             if self.label == None:
#                 return super(GameState, self).__str__()
#             return self.label

class GUI:
    elementSize = 50
    gridBorder = 3
    gridColor = "#000000"
    p1Color = "#FF0000"
    p2Color = "#FFFF00"
    backgroundColor = "#add8e6"
    gameOn = False

    def __init__(self, master):
        self.master = master

        master.title('Connect Four')

        label = Label(master, text="Connect Four", font=("Times New Roman", 50))
        label.grid(row=0,column=1)

        player1label = Label(master,text="If Player 1 is Computer")
        player2label = Label(master,text="If Player 2 is Computer")
        player1button1 = Button(master,text="Click Here!", command=self.cpuDrop1)
        player2button1 = Button(master,text="Click Here!",command=self.cpuDrop2)

        player1label.grid(row=2,column=0,)
        player2label.grid(row=2,column=2)
        player1button1.grid(row=3,column=0,)
        player2button1.grid(row=3,column=2)

        button = Button(master, text="New Game!", command=self._newGameButton)
        button.grid(row=3,column=1)

        self.canvas = Canvas(master, width=200, height=50, background=self.backgroundColor, highlightthickness=0)
        self.canvas.grid(row=5,column=1)

        self.currentPlayerVar = StringVar(self.master, value="")
        self.currentPlayerLabel = Label(self.master, textvariable=self.currentPlayerVar, anchor=W)
        self.currentPlayerLabel.grid(row=6,column=1)

        self.canvas.bind('<Button-1>', self._canvasClick)
        self.newGame()


    def cpuDrop1(self):
        if(self.gameState.first_player == True):
            if not self.gameOn: return
            if self.gameState.game_over: return
            self.adrop(self)
            self.master.update()
            self.drawGrid()
            self.draw()
            self._updateCurrentPlayer()


            if self.gameState.game_over:
                x = self.canvas.winfo_width() // 2
                y = self.canvas.winfo_height() // 2
                if self.gameState.game_over == 'draw':
                    t = 'DRAW!'
                else:
                    winner = self.p1 if self.gameState.first_player else self.p2
                    t = winner + ' won!'
                self.canvas.create_text(x, y, text=t, font=("Helvetica", 32), fill="#333")

    def cpuDrop2(self):
        if(self.gameState.first_player == False):
            if not self.gameOn: return
            if self.gameState.game_over: return

            self.bdrop(self)
            self.master.update()
            self.drawGrid()
            self.draw()
            self._updateCurrentPlayer()


            if self.gameState.game_over:
                x = self.canvas.winfo_width() // 2
                y = self.canvas.winfo_height() // 2
                if self.gameState.game_over == 'draw':
                    t = 'DRAW!'
                else:
                    winner = self.p1 if self.gameState.first_player else self.p2
                    t = winner + ' won!'
                self.canvas.create_text(x, y, text=t, font=("Helvetica", 32), fill="#333")

    def draw(self):
        for c in range(self.gameState.size['c']):
            for r in range(self.gameState.size['r']):
                if r >= len(self.gameState.grid[c]): continue

                x0 = c * self.elementSize
                y0 = r * self.elementSize
                x1 = (c + 1) * self.elementSize
                y1 = (r + 1) * self.elementSize
                fill = self.p1Color if self.gameState.grid[c][r] == self.gameState.players[True] else self.p2Color
                self.canvas.create_oval(x0 + 2,
                                        self.canvas.winfo_height() - (y0 + 2),
                                        x1 - 2,
                                        self.canvas.winfo_height() - (y1 - 2),
                                        fill=fill, outline=self.gridColor)

    def drawGrid(self):
        x0, x1 = 0, self.canvas.winfo_width()
        for r in range(1, self.gameState.size['r']):
            y = r * self.elementSize
            self.canvas.create_line(x0, y, x1, y, fill=self.gridColor)

        y0, y1 = 0, self.canvas.winfo_height()
        for c in range(1, self.gameState.size['c']):
            x = c * self.elementSize
            self.canvas.create_line(x, y0, x, y1, fill=self.gridColor)

    # def drop(self, column):
    #     return self.gameState.drop(column)

    def drop(self, column):
            return self.gameState.drop(column)

    def adrop(self,column):
        if(self.gameState.first_player):

            #print(test)
            print(column.gameState.grid)
            guess = randint(0,6)
            return self.gameState.drop(guess)
        else:
            return self.gameState.drop(column)

    def bdrop(self, column):
        if(self.gameState.first_player):
         #   self.gameState.grid
         #   print(column.gameState.grid)

            return self.gameState.drop(column)

        else:
            for x in range(0,7):
                for y in range(0,len(column.gameState.grid[x])):
                    print(column.gameState.grid[x][y])
                #d = {column.gameState.grid[x], x}

                #print(column.gameState.grid[x])
            #print(b)
            #guess = randint(0, 6)
            # guess = g.utility(self, self.gameState, self.currentPlayerLabel)
            guess = games.alphabeta_search(self.gameState, self.game, 1)
           # print(d)
          #  print(column.gameState.grid)
            return self.gameState.drop(guess)




    def newGame(self):
        self.p1 = 'Player 1'
        self.p2 = 'Player 2'
        columns = 7
        rows = 6

        self.gameState = ConnectFour.ConnectFour(columns=columns, rows=rows)
        self.game = ConnectFour.C4Game(self.gameState)

        self.canvas.delete(ALL)
        self.canvas.config(width=(self.elementSize) * self.gameState.size['c'],
                           height=(self.elementSize) * self.gameState.size['r'])
        self.master.update()
        self.drawGrid()
        self.draw()

        self._updateCurrentPlayer()

        self.gameOn = True

    def _updateCurrentPlayer(self):
        p = self.p1 if self.gameState.first_player else self.p2
        self.currentPlayerVar.set('Current player: ' + p)

    def _canvasClick(self, event):
        if not self.gameOn: return
        if self.gameState.game_over: return

        c = event.x // self.elementSize

        if (0 <= c < self.gameState.size['c']):
            self.drop(c)
            self.draw()
            self._updateCurrentPlayer()

        if self.gameState.game_over:
            x = self.canvas.winfo_width() // 2
            y = self.canvas.winfo_height() // 2
            if self.gameState.game_over == 'draw':
                t = 'DRAW!'
            else:
                winner = self.p1 if self.gameState.first_player else self.p2
                t = winner + ' won!'
          #  self.canvas.create_text(x, y, text=t, font=("Times New Roman", 42), fill="#333")
            self.canvas.create_text(175, y-120, text=t, font=("Times New Roman", 42), fill="#333")

    def _newGameButton(self):
        self.newGame()

    def check_win(self, board):
        if board[0] == 0 and board[1] == 0 and board[2] == 0:
            return 1
        return 0



root = Tk()

#root.configure(background="purple")
app = GUI(root)
root.wm_iconbitmap('4.ico')
root.mainloop()
