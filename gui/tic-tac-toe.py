from tkinter import *
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from games import minimax_decision, alphabeta_player, random_player, TicTacToe
# "gen_state" can be used to generate a game state to apply the algorithm
from tests.test_games import gen_state

ttt = TicTacToe()
root = None
buttons = []
frames = []
x_pos = []
o_pos = []
count = 0
sym = ""
result = None
choices = None


def create_frames(root):
    """
    This function creates the necessary structure of the game.
    """
    frame1 = Frame(root)
    frame2 = Frame(root)
    frame3 = Frame(root)
    frame4 = Frame(root)
    create_buttons(frame1)
    create_buttons(frame2)
    create_buttons(frame3)
    buttonExit = Button(
        frame4, height=1, width=2,
        text="Exit",
        command=lambda: exit_game(root))
    buttonExit.pack(side=LEFT)
    frame4.pack(side=BOTTOM)
    frame3.pack(side=BOTTOM)
    frame2.pack(side=BOTTOM)
    frame1.pack(side=BOTTOM)
    frames.append(frame1)
    frames.append(frame2)
    frames.append(frame3)
    for x in frames:
        buttons_in_frame = []
        for y in x.winfo_children():
            buttons_in_frame.append(y)
        buttons.append(buttons_in_frame)
    buttonReset = Button(frame4, height=1, width=2,
                         text="Reset", command=lambda: reset_game())
    buttonReset.pack(side=LEFT)


def create_buttons(frame):
    """
    This function creates the buttons to be pressed/clicked during the game.
    """
    button0 = Button(frame, height=2, width=2, text=" ",
                     command=lambda: on_click(button0))
    button0.pack(side=LEFT)
    button1 = Button(frame, height=2, width=2, text=" ",
                     command=lambda: on_click(button1))
    button1.pack(side=LEFT)
    button2 = Button(frame, height=2, width=2, text=" ",
                     command=lambda: on_click(button2))
    button2.pack(side=LEFT)


# TODO: Add a choice option for the user.
def on_click(button):
    """
    This function determines the action of any button.
    """
    global ttt, choices, count, sym, result, x_pos, o_pos

    if count % 2 == 0:
        sym = "X"
    else:
        sym = "O"
    count += 1

    button.config(
        text=sym,
        state='disabled',
        disabledforeground="red")  # For cross

    x, y = get_coordinates(button)
    x += 1
    y += 1
    x_pos.append((x, y))
    state = gen_state(to_move='O', x_positions=x_pos,
                      o_positions=o_pos)
    try:
        choice = choices.get()
        if "Random" in choice:
            a, b = random_player(ttt, state)
        elif "Pro" in choice:
            a, b = minimax_decision(state, ttt)
        else:
            a, b = alphabeta_player(ttt, state)
    except (ValueError, IndexError, TypeError) as e:
        disable_game()
        result.set("It's a draw :|")
        return
    if 1 <= a <= 3 and 1 <= b <= 3:
        o_pos.append((a, b))
        button_to_change = get_button(a - 1, b - 1)
        if count % 2 == 0:  # Used again, will become handy when user is given the choice of turn.
            sym = "X"
        else:
            sym = "O"
        count += 1

        if check_victory(button):
            result.set("You win :)")
            disable_game()
        else:
            button_to_change.config(text=sym, state='disabled',
                                    disabledforeground="black")
            if check_victory(button_to_change):
                result.set("You lose :(")
                disable_game()


# TODO: Replace "check_victory" by "k_in_row" function.
def check_victory(button):
    """
    This function checks various winning conditions of the game.
    """
    # check if previous move caused a win on vertical line
    global buttons
    x, y = get_coordinates(button)
    tt = button['text']
    if buttons[0][y]['text'] == buttons[1][y]['text'] == buttons[2][y]['text'] != " ":
        buttons[0][y].config(text="|" + tt + "|")
        buttons[1][y].config(text="|" + tt + "|")
        buttons[2][y].config(text="|" + tt + "|")
        return True

    # check if previous move caused a win on horizontal line
    if buttons[x][0]['text'] == buttons[x][1]['text'] == buttons[x][2]['text'] != " ":
        buttons[x][0].config(text="--" + tt + "--")
        buttons[x][1].config(text="--" + tt + "--")
        buttons[x][2].config(text="--" + tt + "--")
        return True

    # check if previous move was on the main diagonal and caused a win
    if x == y and buttons[0][0]['text'] == buttons[1][1]['text'] == buttons[2][2]['text'] != " ":
        buttons[0][0].config(text="\\" + tt + "\\")
        buttons[1][1].config(text="\\" + tt + "\\")
        buttons[2][2].config(text="\\" + tt + "\\")
        return True

    # check if previous move was on the secondary diagonal and caused a win
    if x + y \
            == 2 and buttons[0][2]['text'] == buttons[1][1]['text'] == buttons[2][0]['text'] != " ":
        buttons[0][2].config(text="/" + tt + "/")
        buttons[1][1].config(text="/" + tt + "/")
        buttons[2][0].config(text="/" + tt + "/")
        return True

    return False


def get_coordinates(button):
    """
    This function returns the coordinates of the button clicked.
    """
    global buttons
    for x in range(len(buttons)):
        for y in range(len(buttons[x])):
            if buttons[x][y] == button:
                return x, y


def get_button(x, y):
    """
    This function returns the button memory location corresponding to a coordinate.
    """
    global buttons
    return buttons[x][y]


def reset_game():
    """
    This function will reset all the tiles to the initial null value.
    """
    global x_pos, o_pos, frames, count

    count = 0
    x_pos = []
    o_pos = []
    result.set("Your Turn!")
    for x in frames:
        for y in x.winfo_children():
            y.config(text=" ", state='normal')


def disable_game():
    """
    This function deactivates the game after a win, loss or draw.
    """
    global frames
    for x in frames:
        for y in x.winfo_children():
            y.config(state='disabled')


def exit_game(root):
    """
    This function will exit the game by killing the root.
    """
    root.destroy()


def main():
    global result, choices

    root = Tk()
    root.title("TicTacToe")
    root.geometry("150x200")  # Improved the window geometry
    root.resizable(0, 0)  # To remove the maximize window option
    result = StringVar()
    result.set("Your Turn!")
    w = Label(root, textvariable=result)
    w.pack(side=BOTTOM)
    create_frames(root)
    choices = StringVar(root)
    choices.set("Vs Pro")
    menu = OptionMenu(root, choices, "Vs Random", "Vs Pro", "Vs Legend")
    menu.pack()
    root.mainloop()


if __name__ == "__main__":
    main()
