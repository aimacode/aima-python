import os.path
import random
import time
from functools import partial
from tkinter import *

from search import astar_search, EightPuzzle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

root = Tk()

state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
puzzle = EightPuzzle(tuple(state))
solution = None

b = [None] * 9


# TODO: refactor into OOP, remove global variables

def scramble():
    """Scrambles the puzzle starting from the goal state"""

    global state
    global puzzle
    possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    scramble = []
    for _ in range(60):
        scramble.append(random.choice(possible_actions))

    for move in scramble:
        if move in puzzle.actions(state):
            state = list(puzzle.result(state, move))
            puzzle = EightPuzzle(tuple(state))
            create_buttons()


def solve():
    """Solves the puzzle using astar_search"""

    return astar_search(puzzle).solution()


def solve_steps():
    """Solves the puzzle step by step"""

    global puzzle
    global solution
    global state
    solution = solve()
    print(solution)

    for move in solution:
        state = puzzle.result(state, move)
        create_buttons()
        root.update()
        root.after(1, time.sleep(0.75))


def exchange(index):
    """Interchanges the position of the selected tile with the zero tile under certain conditions"""

    global state
    global solution
    global puzzle
    zero_ix = list(state).index(0)
    actions = puzzle.actions(state)
    current_action = ''
    i_diff = index // 3 - zero_ix // 3
    j_diff = index % 3 - zero_ix % 3
    if i_diff == 1:
        current_action += 'DOWN'
    elif i_diff == -1:
        current_action += 'UP'

    if j_diff == 1:
        current_action += 'RIGHT'
    elif j_diff == -1:
        current_action += 'LEFT'

    if abs(i_diff) + abs(j_diff) != 1:
        current_action = ''

    if current_action in actions:
        b[zero_ix].grid_forget()
        b[zero_ix] = Button(root, text=f'{state[index]}', width=6, font=('Helvetica', 40, 'bold'),
                            command=partial(exchange, zero_ix))
        b[zero_ix].grid(row=zero_ix // 3, column=zero_ix % 3, ipady=40)
        b[index].grid_forget()
        b[index] = Button(root, text=None, width=6, font=('Helvetica', 40, 'bold'), command=partial(exchange, index))
        b[index].grid(row=index // 3, column=index % 3, ipady=40)
        state[zero_ix], state[index] = state[index], state[zero_ix]
        puzzle = EightPuzzle(tuple(state))


def create_buttons():
    """Creates dynamic buttons"""

    # TODO: Find a way to use grid_forget() with a for loop for initialization
    b[0] = Button(root, text=f'{state[0]}' if state[0] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 0))
    b[0].grid(row=0, column=0, ipady=40)
    b[1] = Button(root, text=f'{state[1]}' if state[1] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 1))
    b[1].grid(row=0, column=1, ipady=40)
    b[2] = Button(root, text=f'{state[2]}' if state[2] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 2))
    b[2].grid(row=0, column=2, ipady=40)
    b[3] = Button(root, text=f'{state[3]}' if state[3] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 3))
    b[3].grid(row=1, column=0, ipady=40)
    b[4] = Button(root, text=f'{state[4]}' if state[4] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 4))
    b[4].grid(row=1, column=1, ipady=40)
    b[5] = Button(root, text=f'{state[5]}' if state[5] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 5))
    b[5].grid(row=1, column=2, ipady=40)
    b[6] = Button(root, text=f'{state[6]}' if state[6] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 6))
    b[6].grid(row=2, column=0, ipady=40)
    b[7] = Button(root, text=f'{state[7]}' if state[7] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 7))
    b[7].grid(row=2, column=1, ipady=40)
    b[8] = Button(root, text=f'{state[8]}' if state[8] != 0 else None, width=6, font=('Helvetica', 40, 'bold'),
                  command=partial(exchange, 8))
    b[8].grid(row=2, column=2, ipady=40)


def create_static_buttons():
    """Creates scramble and solve buttons"""

    scramble_btn = Button(root, text='Scramble', font=('Helvetica', 30, 'bold'), width=8, command=partial(init))
    scramble_btn.grid(row=3, column=0, ipady=10)
    solve_btn = Button(root, text='Solve', font=('Helvetica', 30, 'bold'), width=8, command=partial(solve_steps))
    solve_btn.grid(row=3, column=2, ipady=10)


def init():
    """Calls necessary functions"""

    global state
    global solution
    state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    scramble()
    create_buttons()
    create_static_buttons()


init()
root.mainloop()
