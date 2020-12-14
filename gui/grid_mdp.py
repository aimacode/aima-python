import os.path
import sys
import tkinter as tk
import tkinter.messagebox
from functools import partial
from tkinter import ttk

import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from mdp import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

matplotlib.use('TkAgg')
style.use('ggplot')

fig = Figure(figsize=(20, 15))
sub = fig.add_subplot(111)
plt.rcParams['axes.grid'] = False

WALL_VALUE = -99999.0
TERM_VALUE = -999999.0

black = '#000'
white = '#fff'
gray2 = '#222'
gray9 = '#999'
grayd = '#ddd'
grayef = '#efefef'
pblue = '#000040'
green8 = '#008080'
green4 = '#004040'

cell_window_mantainer = None


def extents(f):
    """adjusts axis markers for heatmap"""

    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


def display(gridmdp, _height, _width):
    """displays matrix"""

    dialog = tk.Toplevel()
    dialog.wm_title('Values')

    container = tk.Frame(dialog)
    container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            label = ttk.Label(container, text=f'{gridmdp[_height - i - 1][j]:.3f}', font=('Helvetica', 12))
            label.grid(row=i + 1, column=j + 1, padx=3, pady=3)

    dialog.mainloop()


def display_best_policy(_best_policy, _height, _width):
    """displays best policy"""
    dialog = tk.Toplevel()
    dialog.wm_title('Best Policy')

    container = tk.Frame(dialog)
    container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            label = ttk.Label(container, text=_best_policy[i][j], font=('Helvetica', 12, 'bold'))
            label.grid(row=i + 1, column=j + 1, padx=3, pady=3)

    dialog.mainloop()


def initialize_dialogbox(_width, _height, gridmdp, terminals, buttons):
    """creates dialogbox for initialization"""

    dialog = tk.Toplevel()
    dialog.wm_title('Initialize')

    container = tk.Frame(dialog)
    container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    wall = tk.IntVar()
    wall.set(0)
    term = tk.IntVar()
    term.set(0)
    reward = tk.DoubleVar()
    reward.set(0.0)

    label = ttk.Label(container, text='Initialize', font=('Helvetica', 12), anchor=tk.N)
    label.grid(row=0, column=0, columnspan=3, sticky='new', pady=15, padx=5)
    label_reward = ttk.Label(container, text='Reward', font=('Helvetica', 10), anchor=tk.N)
    label_reward.grid(row=1, column=0, columnspan=3, sticky='new', pady=1, padx=5)
    entry_reward = ttk.Entry(container, font=('Helvetica', 10), justify=tk.CENTER, exportselection=0,
                             textvariable=reward)
    entry_reward.grid(row=2, column=0, columnspan=3, sticky='new', pady=5, padx=50)

    rbtn_term = ttk.Radiobutton(container, text='Terminal', variable=term, value=TERM_VALUE)
    rbtn_term.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=160, pady=5)
    rbtn_wall = ttk.Radiobutton(container, text='Wall', variable=wall, value=WALL_VALUE)
    rbtn_wall.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=172, pady=5)

    initialize_widget_disability_checks(_width, _height, gridmdp, terminals, label_reward, entry_reward, rbtn_wall,
                                        rbtn_term)

    btn_apply = ttk.Button(container, text='Apply',
                           command=partial(initialize_update_table, _width, _height, gridmdp, terminals, buttons,
                                           reward, term, wall, label_reward, entry_reward, rbtn_term, rbtn_wall))
    btn_apply.grid(row=5, column=0, sticky='nsew', pady=5, padx=5)
    btn_reset = ttk.Button(container, text='Reset',
                           command=partial(initialize_reset_all, _width, _height, gridmdp, terminals, buttons, reward,
                                           term, wall, label_reward, entry_reward, rbtn_wall, rbtn_term))
    btn_reset.grid(row=5, column=1, sticky='nsew', pady=5, padx=5)
    btn_ok = ttk.Button(container, text='Ok', command=dialog.destroy)
    btn_ok.grid(row=5, column=2, sticky='nsew', pady=5, padx=5)

    dialog.geometry('400x200')
    dialog.mainloop()


def update_table(i, j, gridmdp, terminals, buttons, reward, term, wall, label_reward, entry_reward, rbtn_term,
                 rbtn_wall):
    """functionality for 'apply' button"""
    if wall.get() == WALL_VALUE:
        buttons[i][j].configure(style='wall.TButton')
        buttons[i][j].config(text='Wall')
        label_reward.config(foreground='#999')
        entry_reward.config(state=tk.DISABLED)
        rbtn_term.state(['!focus', '!selected'])
        rbtn_term.config(state=tk.DISABLED)
        gridmdp[i][j] = WALL_VALUE

    elif wall.get() != WALL_VALUE:
        if reward.get() != 0.0:
            gridmdp[i][j] = reward.get()
            buttons[i][j].configure(style='reward.TButton')
            buttons[i][j].config(text=f'R = {reward.get()}')

        if term.get() == TERM_VALUE:
            if (i, j) not in terminals:
                terminals.append((i, j))
            rbtn_wall.state(['!focus', '!selected'])
            rbtn_wall.config(state=tk.DISABLED)

            if gridmdp[i][j] < 0:
                buttons[i][j].configure(style='-term.TButton')

            elif gridmdp[i][j] > 0:
                buttons[i][j].configure(style='+term.TButton')

            elif gridmdp[i][j] == 0.0:
                buttons[i][j].configure(style='=term.TButton')


def initialize_update_table(_width, _height, gridmdp, terminals, buttons, reward, term, wall, label_reward,
                            entry_reward, rbtn_term, rbtn_wall):
    """runs update_table for all cells"""

    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            update_table(i, j, gridmdp, terminals, buttons, reward, term, wall, label_reward, entry_reward, rbtn_term,
                         rbtn_wall)


def reset_all(_height, i, j, gridmdp, terminals, buttons, reward, term, wall, label_reward, entry_reward, rbtn_wall,
              rbtn_term):
    """functionality for reset button"""
    reward.set(0.0)
    term.set(0)
    wall.set(0)
    gridmdp[i][j] = 0.0
    buttons[i][j].configure(style='TButton')
    buttons[i][j].config(text=f'({_height - i - 1}, {j})')

    if (i, j) in terminals:
        terminals.remove((i, j))

    label_reward.config(foreground='#000')
    entry_reward.config(state=tk.NORMAL)
    rbtn_term.config(state=tk.NORMAL)
    rbtn_wall.config(state=tk.NORMAL)
    rbtn_wall.state(['!focus', '!selected'])
    rbtn_term.state(['!focus', '!selected'])


def initialize_reset_all(_width, _height, gridmdp, terminals, buttons, reward, term, wall, label_reward, entry_reward,
                         rbtn_wall, rbtn_term):
    """runs reset_all for all cells"""

    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            reset_all(_height, i, j, gridmdp, terminals, buttons, reward, term, wall, label_reward, entry_reward,
                      rbtn_wall, rbtn_term)


def external_reset(_width, _height, gridmdp, terminals, buttons):
    """reset from edit menu"""
    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            gridmdp[i][j] = 0.0
            buttons[i][j].configure(style='TButton')
            buttons[i][j].config(text=f'({_height - i - 1}, {j})')


def widget_disability_checks(i, j, gridmdp, terminals, label_reward, entry_reward, rbtn_wall, rbtn_term):
    """checks for required state of widgets in dialog boxes"""

    if gridmdp[i][j] == WALL_VALUE:
        label_reward.config(foreground='#999')
        entry_reward.config(state=tk.DISABLED)
        rbtn_term.config(state=tk.DISABLED)
        rbtn_wall.state(['!focus', 'selected'])
        rbtn_term.state(['!focus', '!selected'])

    if (i, j) in terminals:
        rbtn_wall.config(state=tk.DISABLED)
        rbtn_wall.state(['!focus', '!selected'])


def flatten_list(_list):
    """returns a flattened list"""
    return sum(_list, [])


def initialize_widget_disability_checks(_width, _height, gridmdp, terminals, label_reward, entry_reward, rbtn_wall,
                                        rbtn_term):
    """checks for required state of widgets when cells are initialized"""

    bool_walls = [['False'] * max(1, _width) for _ in range(max(1, _height))]
    bool_terms = [['False'] * max(1, _width) for _ in range(max(1, _height))]

    for i in range(max(1, _height)):
        for j in range(max(1, _width)):
            if gridmdp[i][j] == WALL_VALUE:
                bool_walls[i][j] = 'True'

            if (i, j) in terminals:
                bool_terms[i][j] = 'True'

    bool_walls_fl = flatten_list(bool_walls)
    bool_terms_fl = flatten_list(bool_terms)

    if bool_walls_fl.count('True') == len(bool_walls_fl):
        print('`')
        label_reward.config(foreground='#999')
        entry_reward.config(state=tk.DISABLED)
        rbtn_term.config(state=tk.DISABLED)
        rbtn_wall.state(['!focus', 'selected'])
        rbtn_term.state(['!focus', '!selected'])

    if bool_terms_fl.count('True') == len(bool_terms_fl):
        rbtn_wall.config(state=tk.DISABLED)
        rbtn_wall.state(['!focus', '!selected'])
        rbtn_term.state(['!focus', 'selected'])


def dialogbox(i, j, gridmdp, terminals, buttons, _height):
    """creates dialogbox for each cell"""
    global cell_window_mantainer
    if (cell_window_mantainer != None):
        cell_window_mantainer.destroy()

    dialog = tk.Toplevel()
    cell_window_mantainer = dialog
    dialog.wm_title(f'{_height - i - 1}, {j}')

    container = tk.Frame(dialog)
    container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    wall = tk.IntVar()
    wall.set(gridmdp[i][j])
    term = tk.IntVar()
    term.set(TERM_VALUE if (i, j) in terminals else 0.0)
    reward = tk.DoubleVar()
    reward.set(gridmdp[i][j] if gridmdp[i][j] != WALL_VALUE else 0.0)

    label = ttk.Label(container, text=f'Configure cell {_height - i - 1}, {j}', font=('Helvetica', 12), anchor=tk.N)
    label.grid(row=0, column=0, columnspan=3, sticky='new', pady=15, padx=5)
    label_reward = ttk.Label(container, text='Reward', font=('Helvetica', 10), anchor=tk.N)
    label_reward.grid(row=1, column=0, columnspan=3, sticky='new', pady=1, padx=5)
    entry_reward = ttk.Entry(container, font=('Helvetica', 10), justify=tk.CENTER, exportselection=0,
                             textvariable=reward)
    entry_reward.grid(row=2, column=0, columnspan=3, sticky='new', pady=5, padx=50)

    rbtn_term = ttk.Radiobutton(container, text='Terminal', variable=term, value=TERM_VALUE)
    rbtn_term.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=160, pady=5)
    rbtn_wall = ttk.Radiobutton(container, text='Wall', variable=wall, value=WALL_VALUE)
    rbtn_wall.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=172, pady=5)

    widget_disability_checks(i, j, gridmdp, terminals, label_reward, entry_reward, rbtn_wall, rbtn_term)

    btn_apply = ttk.Button(container, text='Apply',
                           command=partial(update_table, i, j, gridmdp, terminals, buttons, reward, term, wall,
                                           label_reward, entry_reward, rbtn_term, rbtn_wall))
    btn_apply.grid(row=5, column=0, sticky='nsew', pady=5, padx=5)
    btn_reset = ttk.Button(container, text='Reset',
                           command=partial(reset_all, _height, i, j, gridmdp, terminals, buttons, reward, term, wall,
                                           label_reward, entry_reward, rbtn_wall, rbtn_term))
    btn_reset.grid(row=5, column=1, sticky='nsew', pady=5, padx=5)
    btn_ok = ttk.Button(container, text='Ok', command=dialog.destroy)
    btn_ok.grid(row=5, column=2, sticky='nsew', pady=5, padx=5)

    dialog.geometry('400x200')
    dialog.mainloop()


class MDPapp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, 'Grid MDP')
        self.shared_data = {
            'height': tk.IntVar(),
            'width': tk.IntVar()}
        self.shared_data['height'].set(1)
        self.shared_data['width'].set(1)
        self.container = tk.Frame(self)
        self.container.pack(side='top', fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        self.menu_bar = tk.Menu(self.container)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label='Exit', command=self.exit)
        self.menu_bar.add_cascade(label='File', menu=self.file_menu)

        self.edit_menu = tk.Menu(self.menu_bar, tearoff=1)
        self.edit_menu.add_command(label='Reset', command=self.master_reset)
        self.edit_menu.add_command(label='Initialize', command=self.initialize)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label='View matrix', command=self.view_matrix)
        self.edit_menu.add_command(label='View terminals', command=self.view_terminals)
        self.menu_bar.add_cascade(label='Edit', menu=self.edit_menu)
        self.menu_bar.entryconfig('Edit', state=tk.DISABLED)

        self.build_menu = tk.Menu(self.menu_bar, tearoff=1)
        self.build_menu.add_command(label='Build and Run', command=self.build)
        self.menu_bar.add_cascade(label='Build', menu=self.build_menu)
        self.menu_bar.entryconfig('Build', state=tk.DISABLED)
        tk.Tk.config(self, menu=self.menu_bar)

        for F in (HomePage, BuildMDP, SolveMDP):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(HomePage)

    def placeholder_function(self):
        """placeholder function"""

        print('Not supported yet!')

    def exit(self):
        """function to exit"""
        if tkinter.messagebox.askokcancel('Exit?', 'All changes will be lost'):
            quit()

    def new(self):
        """function to create new GridMDP"""

        self.master_reset()
        build_page = self.get_page(BuildMDP)
        build_page.gridmdp = None
        build_page.terminals = None
        build_page.buttons = None
        self.show_frame(HomePage)

    def get_page(self, page_class):
        """returns pages from stored frames"""
        return self.frames[page_class]

    def view_matrix(self):
        """prints current matrix to console"""

        build_page = self.get_page(BuildMDP)
        _height = self.shared_data['height'].get()
        _width = self.shared_data['width'].get()
        print(build_page.gridmdp)
        display(build_page.gridmdp, _height, _width)

    def view_terminals(self):
        """prints current terminals to console"""
        build_page = self.get_page(BuildMDP)
        print('Terminals', build_page.terminals)

    def initialize(self):
        """calls initialize from BuildMDP"""

        build_page = self.get_page(BuildMDP)
        build_page.initialize()

    def master_reset(self):
        """calls master_reset from BuildMDP"""
        build_page = self.get_page(BuildMDP)
        build_page.master_reset()

    def build(self):
        """runs specified mdp solving algorithm"""

        frame = SolveMDP(self.container, self)
        self.frames[SolveMDP] = frame
        frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame(SolveMDP)
        build_page = self.get_page(BuildMDP)
        gridmdp = build_page.gridmdp
        terminals = build_page.terminals
        solve_page = self.get_page(SolveMDP)
        _height = self.shared_data['height'].get()
        _width = self.shared_data['width'].get()
        solve_page.create_graph(gridmdp, terminals, _height, _width)

    def show_frame(self, controller, cb=False):
        """shows specified frame and optionally runs create_buttons"""
        if cb:
            build_page = self.get_page(BuildMDP)
            build_page.create_buttons()
        frame = self.frames[controller]
        frame.tkraise()


class HomePage(tk.Frame):

    def __init__(self, parent, controller):
        """HomePage constructor"""

        tk.Frame.__init__(self, parent)
        self.controller = controller
        frame1 = tk.Frame(self)
        frame1.pack(side=tk.TOP)
        frame3 = tk.Frame(self)
        frame3.pack(side=tk.TOP)
        frame4 = tk.Frame(self)
        frame4.pack(side=tk.TOP)
        frame2 = tk.Frame(self)
        frame2.pack(side=tk.TOP)

        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TButton', background=grayd, padding=0)
        s.configure('wall.TButton', background=gray2, foreground=white)
        s.configure('reward.TButton', background=gray9)
        s.configure('+term.TButton', background=green8)
        s.configure('-term.TButton', background=pblue, foreground=white)
        s.configure('=term.TButton', background=green4)

        label = ttk.Label(frame1, text='GridMDP builder', font=('Helvetica', 18, 'bold'), background=grayef)
        label.pack(pady=75, padx=50, side=tk.TOP)

        ec_btn = ttk.Button(frame3, text='Empty cells', width=20)
        ec_btn.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        ec_btn.configure(style='TButton')

        w_btn = ttk.Button(frame3, text='Walls', width=20)
        w_btn.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        w_btn.configure(style='wall.TButton')

        r_btn = ttk.Button(frame3, text='Rewards', width=20)
        r_btn.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        r_btn.configure(style='reward.TButton')

        term_p = ttk.Button(frame3, text='Positive terminals', width=20)
        term_p.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        term_p.configure(style='+term.TButton')

        term_z = ttk.Button(frame3, text='Neutral terminals', width=20)
        term_z.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        term_z.configure(style='=term.TButton')

        term_n = ttk.Button(frame3, text='Negative terminals', width=20)
        term_n.pack(pady=0, padx=0, side=tk.LEFT, ipady=10)
        term_n.configure(style='-term.TButton')

        label = ttk.Label(frame4, text='Dimensions', font=('Verdana', 14), background=grayef)
        label.pack(pady=15, padx=10, side=tk.TOP)
        entry_h = tk.Entry(frame2, textvariable=self.controller.shared_data['height'], font=('Verdana', 10), width=3,
                           justify=tk.CENTER)
        entry_h.pack(pady=10, padx=10, side=tk.LEFT)
        label_x = ttk.Label(frame2, text='X', font=('Verdana', 10), background=grayef)
        label_x.pack(pady=10, padx=4, side=tk.LEFT)
        entry_w = tk.Entry(frame2, textvariable=self.controller.shared_data['width'], font=('Verdana', 10), width=3,
                           justify=tk.CENTER)
        entry_w.pack(pady=10, padx=10, side=tk.LEFT)
        button = ttk.Button(self, text='Build a GridMDP', command=lambda: controller.show_frame(BuildMDP, cb=True))
        button.pack(pady=10, padx=10, side=tk.TOP, ipadx=20, ipady=10)
        button.configure(style='reward.TButton')


class BuildMDP(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.frame = tk.Frame(self)
        self.frame.pack()
        self.controller = controller

    def create_buttons(self):
        """creates interactive cells to build MDP"""
        _height = self.controller.shared_data['height'].get()
        _width = self.controller.shared_data['width'].get()
        self.controller.menu_bar.entryconfig('Edit', state=tk.NORMAL)
        self.controller.menu_bar.entryconfig('Build', state=tk.NORMAL)
        self.gridmdp = [[0.0] * max(1, _width) for _ in range(max(1, _height))]
        self.buttons = [[None] * max(1, _width) for _ in range(max(1, _height))]
        self.terminals = []

        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TButton', background=grayd, padding=0)
        s.configure('wall.TButton', background=gray2, foreground=white)
        s.configure('reward.TButton', background=gray9)
        s.configure('+term.TButton', background=green8)
        s.configure('-term.TButton', background=pblue, foreground=white)
        s.configure('=term.TButton', background=green4)

        for i in range(max(1, _height)):
            for j in range(max(1, _width)):
                self.buttons[i][j] = ttk.Button(self.frame, text=f'({_height - i - 1}, {j})',
                                                width=int(196 / max(1, _width)),
                                                command=partial(dialogbox, i, j, self.gridmdp, self.terminals,
                                                                self.buttons, _height))
                self.buttons[i][j].grid(row=i, column=j, ipady=int(336 / max(1, _height)) - 12)

    def initialize(self):
        """runs initialize_dialogbox"""

        _height = self.controller.shared_data['height'].get()
        _width = self.controller.shared_data['width'].get()
        initialize_dialogbox(_width, _height, self.gridmdp, self.terminals, self.buttons)

    def master_reset(self):
        """runs external reset"""
        _height = self.controller.shared_data['height'].get()
        _width = self.controller.shared_data['width'].get()
        if tkinter.messagebox.askokcancel('Reset', 'Are you sure you want to reset all cells?'):
            external_reset(_width, _height, self.gridmdp, self.terminals, self.buttons)


class SolveMDP(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.frame = tk.Frame(self)
        self.frame.pack()
        self.controller = controller
        self.terminated = False
        self.iterations = 0
        self.epsilon = 0.001
        self.delta = 0

    def process_data(self, terminals, _height, _width, gridmdp):
        """preprocess variables"""

        flipped_terminals = []

        for terminal in terminals:
            flipped_terminals.append((terminal[1], _height - terminal[0] - 1))

        grid_to_solve = [[0.0] * max(1, _width) for _ in range(max(1, _height))]
        grid_to_show = [[0.0] * max(1, _width) for _ in range(max(1, _height))]

        for i in range(max(1, _height)):
            for j in range(max(1, _width)):
                if gridmdp[i][j] == WALL_VALUE:
                    grid_to_show[i][j] = 0.0
                    grid_to_solve[i][j] = None

                else:
                    grid_to_show[i][j] = grid_to_solve[i][j] = gridmdp[i][j]

        return flipped_terminals, grid_to_solve, np.flipud(grid_to_show)

    def create_graph(self, gridmdp, terminals, _height, _width):
        """creates canvas and initializes value_iteration_parameters"""
        self._height = _height
        self._width = _width
        self.controller.menu_bar.entryconfig('Edit', state=tk.DISABLED)
        self.controller.menu_bar.entryconfig('Build', state=tk.DISABLED)

        self.terminals, self.gridmdp, self.grid_to_show = self.process_data(terminals, _height, _width, gridmdp)
        self.sequential_decision_environment = GridMDP(self.gridmdp, terminals=self.terminals)

        self.initialize_value_iteration_parameters(self.sequential_decision_environment)

        self.canvas = FigureCanvasTkAgg(fig, self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.anim = animation.FuncAnimation(fig, self.animate_graph, interval=50)
        self.canvas.show()

    def animate_graph(self, i):
        """performs value iteration and animates graph"""

        # cmaps to use: bone_r, Oranges, inferno, BrBG, copper
        self.iterations += 1
        x_interval = max(2, len(self.gridmdp[0]))
        y_interval = max(2, len(self.gridmdp))
        x = np.linspace(0, len(self.gridmdp[0]) - 1, x_interval)
        y = np.linspace(0, len(self.gridmdp) - 1, y_interval)

        sub.clear()
        sub.imshow(self.grid_to_show, cmap='BrBG', aspect='auto', interpolation='none', extent=extents(x) + extents(y),
                   origin='lower')
        fig.tight_layout()

        U = self.U1.copy()

        for s in self.sequential_decision_environment.states:
            self.U1[s] = self.R(s) + self.gamma * max(
                [sum([p * U[s1] for (p, s1) in self.T(s, a)]) for a in self.sequential_decision_environment.actions(s)])
            self.delta = max(self.delta, abs(self.U1[s] - U[s]))

        self.grid_to_show = grid_to_show = [[0.0] * max(1, self._width) for _ in range(max(1, self._height))]
        for k, v in U.items():
            self.grid_to_show[k[1]][k[0]] = v

        if (self.delta < self.epsilon * (1 - self.gamma) / self.gamma) or (
                self.iterations > 60) and self.terminated is False:
            self.terminated = True
            display(self.grid_to_show, self._height, self._width)

            pi = best_policy(self.sequential_decision_environment,
                             value_iteration(self.sequential_decision_environment, .01))
            display_best_policy(self.sequential_decision_environment.to_arrows(pi), self._height, self._width)

        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def initialize_value_iteration_parameters(self, mdp):
        """initializes value_iteration parameters"""
        self.U1 = {s: 0 for s in mdp.states}
        self.R, self.T, self.gamma = mdp.R, mdp.T, mdp.gamma

    def value_iteration_metastep(self, mdp, iterations=20):
        """runs value_iteration"""

        U_over_time = []
        U1 = {s: 0 for s in mdp.states}
        R, T, gamma = mdp.R, mdp.T, mdp.gamma

        for _ in range(iterations):
            U = U1.copy()

            for s in mdp.states:
                U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)]) for a in mdp.actions(s)])

            U_over_time.append(U)
        return U_over_time


if __name__ == '__main__':
    app = MDPapp()
    app.geometry('1280x720')
    app.mainloop()
