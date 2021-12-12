import time
from collections import defaultdict
from inspect import getsource

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from IPython.display import display
from PIL import Image
from matplotlib import lines

from games import TicTacToe, alpha_beta_player, random_player, Fig52Extended
from learning import DataSet
from logic import parse_definite_clause, standardize_variables, unify_mm, subst
from search import GraphProblem, romania_map


# ______________________________________________________________________________
# Magic Words


def pseudocode(algorithm):
    """Print the pseudocode for the given algorithm."""
    from urllib.request import urlopen
    from IPython.display import Markdown

    algorithm = algorithm.replace(' ', '-')
    url = "https://raw.githubusercontent.com/aimacode/aima-pseudocode/master/md/{}.md".format(algorithm)
    f = urlopen(url)
    md = f.read().decode('utf-8')
    md = md.split('\n', 1)[-1].strip()
    md = '#' + md
    return Markdown(md)


def psource(*functions):
    """Print the source code for the given function(s)."""
    source_code = '\n\n'.join(getsource(fn) for fn in functions)
    try:
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight

        display(HTML(highlight(source_code, PythonLexer(), HtmlFormatter(full=True))))

    except ImportError:
        print(source_code)


# ______________________________________________________________________________
# Iris Visualization


def show_iris(i=0, j=1, k=2):
    """Plots the iris dataset in a 3D plot.
    The three axes are given by i, j and k,
    which correspond to three of the four iris features."""

    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    iris = DataSet(name="iris")
    buckets = iris.split_values_by_classes()

    features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    f1, f2, f3 = features[i], features[j], features[k]

    a_setosa = [v[i] for v in buckets["setosa"]]
    b_setosa = [v[j] for v in buckets["setosa"]]
    c_setosa = [v[k] for v in buckets["setosa"]]

    a_virginica = [v[i] for v in buckets["virginica"]]
    b_virginica = [v[j] for v in buckets["virginica"]]
    c_virginica = [v[k] for v in buckets["virginica"]]

    a_versicolor = [v[i] for v in buckets["versicolor"]]
    b_versicolor = [v[j] for v in buckets["versicolor"]]
    c_versicolor = [v[k] for v in buckets["versicolor"]]

    for c, m, sl, sw, pl in [('b', 's', a_setosa, b_setosa, c_setosa),
                             ('g', '^', a_virginica, b_virginica, c_virginica),
                             ('r', 'o', a_versicolor, b_versicolor, c_versicolor)]:
        ax.scatter(sl, sw, pl, c=c, marker=m)

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)

    plt.show()


# ______________________________________________________________________________
# MNIST


def load_MNIST(path="aima-data/MNIST/Digits", fashion=False):
    import os, struct
    import array
    import numpy as np

    if fashion:
        path = "aima-data/MNIST/Fashion"

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    train_img_file = open(os.path.join(path, "train-images-idx3-ubyte"), "rb")
    train_lbl_file = open(os.path.join(path, "train-labels-idx1-ubyte"), "rb")
    test_img_file = open(os.path.join(path, "t10k-images-idx3-ubyte"), "rb")
    test_lbl_file = open(os.path.join(path, 't10k-labels-idx1-ubyte'), "rb")

    magic_nr, tr_size, tr_rows, tr_cols = struct.unpack(">IIII", train_img_file.read(16))
    tr_img = array.array("B", train_img_file.read())
    train_img_file.close()
    magic_nr, tr_size = struct.unpack(">II", train_lbl_file.read(8))
    tr_lbl = array.array("b", train_lbl_file.read())
    train_lbl_file.close()

    magic_nr, te_size, te_rows, te_cols = struct.unpack(">IIII", test_img_file.read(16))
    te_img = array.array("B", test_img_file.read())
    test_img_file.close()
    magic_nr, te_size = struct.unpack(">II", test_lbl_file.read(8))
    te_lbl = array.array("b", test_lbl_file.read())
    test_lbl_file.close()

    # print(len(tr_img), len(tr_lbl), tr_size)
    # print(len(te_img), len(te_lbl), te_size)

    train_img = np.zeros((tr_size, tr_rows * tr_cols), dtype=np.int16)
    train_lbl = np.zeros((tr_size,), dtype=np.int8)
    for i in range(tr_size):
        train_img[i] = np.array(tr_img[i * tr_rows * tr_cols: (i + 1) * tr_rows * tr_cols]).reshape((tr_rows * te_cols))
        train_lbl[i] = tr_lbl[i]

    test_img = np.zeros((te_size, te_rows * te_cols), dtype=np.int16)
    test_lbl = np.zeros((te_size,), dtype=np.int8)
    for i in range(te_size):
        test_img[i] = np.array(te_img[i * te_rows * te_cols: (i + 1) * te_rows * te_cols]).reshape((te_rows * te_cols))
        test_lbl[i] = te_lbl[i]

    return (train_img, train_lbl, test_img, test_lbl)


digit_classes = [str(i) for i in range(10)]
fashion_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def show_MNIST(labels, images, samples=8, fashion=False):
    if not fashion:
        classes = digit_classes
    else:
        classes = fashion_classes

    num_classes = len(classes)

    for y, cls in enumerate(classes):
        idxs = np.nonzero([i == y for i in labels])
        idxs = np.random.choice(idxs[0], samples, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples, num_classes, plt_idx)
            plt.imshow(images[idx].reshape((28, 28)))
            plt.axis("off")
            if i == 0:
                plt.title(cls)

    plt.show()


def show_ave_MNIST(labels, images, fashion=False):
    if not fashion:
        item_type = "Digit"
        classes = digit_classes
    else:
        item_type = "Apparel"
        classes = fashion_classes

    num_classes = len(classes)

    for y, cls in enumerate(classes):
        idxs = np.nonzero([i == y for i in labels])
        print(item_type, y, ":", len(idxs[0]), "images.")

        ave_img = np.mean(np.vstack([images[i] for i in idxs[0]]), axis=0)
        # print(ave_img.shape)

        plt.subplot(1, num_classes, y + 1)
        plt.imshow(ave_img.reshape((28, 28)))
        plt.axis("off")
        plt.title(cls)

    plt.show()


# ______________________________________________________________________________
# MDP


def make_plot_grid_step_function(columns, rows, U_over_time):
    """ipywidgets interactive function supports single parameter as input.
    This function creates and return such a function by taking as input
    other parameters."""

    def plot_grid_step(iteration):
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse()  # output like book
        fig = plt.imshow(grid, cmap=plt.cm.bwr, interpolation='nearest')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        for col in range(len(grid)):
            for row in range(len(grid[0])):
                magic = grid[col][row]
                fig.axes.text(row, col, "{0:.2f}".format(magic), va='center', ha='center')

        plt.show()

    return plot_grid_step


def make_visualize(slider):
    """Takes an input a sliderand returns callback function
    for timer and animation."""

    def visualize_callback(visualize, time_step):
        if visualize is True:
            for i in range(slider.min, slider.max + 1):
                slider.value = i
                time.sleep(float(time_step))

    return visualize_callback


# ______________________________________________________________________________


_canvas = """
<script type="text/javascript" src="./js/canvas.js"></script>
<div>
<canvas id="{0}" width="{1}" height="{2}" style="background:rgba(158, 167, 184, 0.2);" onclick='click_callback(this, event, "{3}")'></canvas>
</div>

<script> var {0}_canvas_object = new Canvas("{0}");</script>
"""  # noqa


class Canvas:
    """Inherit from this class to manage the HTML canvas element in jupyter notebooks.
    To create an object of this class any_name_xyz = Canvas("any_name_xyz")
    The first argument given must be the name of the object being created.
    IPython must be able to reference the variable name that is being passed."""

    def __init__(self, varname, width=800, height=600, cid=None):
        self.name = varname
        self.cid = cid or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.cid, self.width, self.height, self.name)
        self.exec_list = []
        display_html(self.html)

    def mouse_click(self, x, y):
        """Override this method to handle mouse click at position (x, y)"""
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def execute(self, exec_str):
        """Stores the command to be executed to a list which is used later during update()"""
        if not isinstance(exec_str, str):
            print("Invalid execution argument:", exec_str)
            self.alert("Received invalid execution command format")
        prefix = "{0}_canvas_object.".format(self.cid)
        self.exec_list.append(prefix + exec_str + ';')

    def fill(self, r, g, b):
        """Changes the fill color to a color in rgb format"""
        self.execute("fill({0}, {1}, {2})".format(r, g, b))

    def stroke(self, r, g, b):
        """Changes the colors of line/strokes to rgb"""
        self.execute("stroke({0}, {1}, {2})".format(r, g, b))

    def strokeWidth(self, w):
        """Changes the width of lines/strokes to 'w' pixels"""
        self.execute("strokeWidth({0})".format(w))

    def rect(self, x, y, w, h):
        """Draw a rectangle with 'w' width, 'h' height and (x, y) as the top-left corner"""
        self.execute("rect({0}, {1}, {2}, {3})".format(x, y, w, h))

    def rect_n(self, xn, yn, wn, hn):
        """Similar to rect(), but the dimensions are normalized to fall between 0 and 1"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        w = round(wn * self.width)
        h = round(hn * self.height)
        self.rect(x, y, w, h)

    def line(self, x1, y1, x2, y2):
        """Draw a line from (x1, y1) to (x2, y2)"""
        self.execute("line({0}, {1}, {2}, {3})".format(x1, y1, x2, y2))

    def line_n(self, x1n, y1n, x2n, y2n):
        """Similar to line(), but the dimensions are normalized to fall between 0 and 1"""
        x1 = round(x1n * self.width)
        y1 = round(y1n * self.height)
        x2 = round(x2n * self.width)
        y2 = round(y2n * self.height)
        self.line(x1, y1, x2, y2)

    def arc(self, x, y, r, start, stop):
        """Draw an arc with (x, y) as centre, 'r' as radius from angles 'start' to 'stop'"""
        self.execute("arc({0}, {1}, {2}, {3}, {4})".format(x, y, r, start, stop))

    def arc_n(self, xn, yn, rn, start, stop):
        """Similar to arc(), but the dimensions are normalized to fall between 0 and 1
        The normalizing factor for radius is selected between width and height by
        seeing which is smaller."""
        x = round(xn * self.width)
        y = round(yn * self.height)
        r = round(rn * min(self.width, self.height))
        self.arc(x, y, r, start, stop)

    def clear(self):
        """Clear the HTML canvas"""
        self.execute("clear()")

    def font(self, font):
        """Changes the font of text"""
        self.execute('font("{0}")'.format(font))

    def text(self, txt, x, y, fill=True):
        """Display a text at (x, y)"""
        if fill:
            self.execute('fill_text("{0}", {1}, {2})'.format(txt, x, y))
        else:
            self.execute('stroke_text("{0}", {1}, {2})'.format(txt, x, y))

    def text_n(self, txt, xn, yn, fill=True):
        """Similar to text(), but with normalized coordinates"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        self.text(txt, x, y, fill)

    def alert(self, message):
        """Immediately display an alert"""
        display_html('<script>alert("{0}")</script>'.format(message))

    def update(self):
        """Execute the JS code to execute the commands queued by execute()"""
        exec_code = "<script>\n" + '\n'.join(self.exec_list) + "\n</script>"
        self.exec_list = []
        display_html(exec_code)


def display_html(html_string):
    display(HTML(html_string))


################################################################################


class Canvas_TicTacToe(Canvas):
    """Play a 3x3 TicTacToe game on HTML canvas"""

    def __init__(self, varname, player_1='human', player_2='random',
                 width=300, height=350, cid=None):
        valid_players = ('human', 'random', 'alpha_beta')
        if player_1 not in valid_players or player_2 not in valid_players:
            raise TypeError("Players must be one of {}".format(valid_players))
        super().__init__(varname, width, height, cid)
        self.ttt = TicTacToe()
        self.state = self.ttt.initial
        self.turn = 0
        self.strokeWidth(5)
        self.players = (player_1, player_2)
        self.font("20px Arial")
        self.draw_board()

    def mouse_click(self, x, y):
        player = self.players[self.turn]
        if self.ttt.terminal_test(self.state):
            if 0.55 <= x / self.width <= 0.95 and 6 / 7 <= y / self.height <= 6 / 7 + 1 / 8:
                self.state = self.ttt.initial
                self.turn = 0
                self.draw_board()
            return

        if player == 'human':
            x, y = int(3 * x / self.width) + 1, int(3 * y / (self.height * 6 / 7)) + 1
            if (x, y) not in self.ttt.actions(self.state):
                # Invalid move
                return
            move = (x, y)
        elif player == 'alpha_beta':
            move = alpha_beta_player(self.ttt, self.state)
        else:
            move = random_player(self.ttt, self.state)
        self.state = self.ttt.result(self.state, move)
        self.turn ^= 1
        self.draw_board()

    def draw_board(self):
        self.clear()
        self.stroke(0, 0, 0)
        offset = 1 / 20
        self.line_n(0 + offset, (1 / 3) * 6 / 7, 1 - offset, (1 / 3) * 6 / 7)
        self.line_n(0 + offset, (2 / 3) * 6 / 7, 1 - offset, (2 / 3) * 6 / 7)
        self.line_n(1 / 3, (0 + offset) * 6 / 7, 1 / 3, (1 - offset) * 6 / 7)
        self.line_n(2 / 3, (0 + offset) * 6 / 7, 2 / 3, (1 - offset) * 6 / 7)

        board = self.state.board
        for mark in board:
            if board[mark] == 'X':
                self.draw_x(mark)
            elif board[mark] == 'O':
                self.draw_o(mark)
        if self.ttt.terminal_test(self.state):
            # End game message
            utility = self.ttt.utility(self.state, self.ttt.to_move(self.ttt.initial))
            if utility == 0:
                self.text_n('Game Draw!', offset, 6 / 7 + offset)
            else:
                self.text_n('Player {} wins!'.format("XO"[utility < 0]), offset, 6 / 7 + offset)
                # Find the 3 and draw a line
                self.stroke([255, 0][self.turn], [0, 255][self.turn], 0)
                for i in range(3):
                    if all([(i + 1, j + 1) in self.state.board for j in range(3)]) and \
                            len({self.state.board[(i + 1, j + 1)] for j in range(3)}) == 1:
                        self.line_n(i / 3 + 1 / 6, offset * 6 / 7, i / 3 + 1 / 6, (1 - offset) * 6 / 7)
                    if all([(j + 1, i + 1) in self.state.board for j in range(3)]) and \
                            len({self.state.board[(j + 1, i + 1)] for j in range(3)}) == 1:
                        self.line_n(offset, (i / 3 + 1 / 6) * 6 / 7, 1 - offset, (i / 3 + 1 / 6) * 6 / 7)
                if all([(i + 1, i + 1) in self.state.board for i in range(3)]) and \
                        len({self.state.board[(i + 1, i + 1)] for i in range(3)}) == 1:
                    self.line_n(offset, offset * 6 / 7, 1 - offset, (1 - offset) * 6 / 7)
                if all([(i + 1, 3 - i) in self.state.board for i in range(3)]) and \
                        len({self.state.board[(i + 1, 3 - i)] for i in range(3)}) == 1:
                    self.line_n(offset, (1 - offset) * 6 / 7, 1 - offset, offset * 6 / 7)
            # restart button
            self.fill(0, 0, 255)
            self.rect_n(0.5 + offset, 6 / 7, 0.4, 1 / 8)
            self.fill(0, 0, 0)
            self.text_n('Restart', 0.5 + 2 * offset, 13 / 14)
        else:  # Print which player's turn it is
            self.text_n("Player {}'s move({})".format("XO"[self.turn], self.players[self.turn]),
                        offset, 6 / 7 + offset)

        self.update()

    def draw_x(self, position):
        self.stroke(0, 255, 0)
        x, y = [i - 1 for i in position]
        offset = 1 / 15
        self.line_n(x / 3 + offset, (y / 3 + offset) * 6 / 7, x / 3 + 1 / 3 - offset, (y / 3 + 1 / 3 - offset) * 6 / 7)
        self.line_n(x / 3 + 1 / 3 - offset, (y / 3 + offset) * 6 / 7, x / 3 + offset, (y / 3 + 1 / 3 - offset) * 6 / 7)

    def draw_o(self, position):
        self.stroke(255, 0, 0)
        x, y = [i - 1 for i in position]
        self.arc_n(x / 3 + 1 / 6, (y / 3 + 1 / 6) * 6 / 7, 1 / 9, 0, 360)


class Canvas_min_max(Canvas):
    """MinMax for Fig52Extended on HTML canvas"""

    def __init__(self, varname, util_list, width=800, height=600, cid=None):
        super().__init__(varname, width, height, cid)
        self.utils = {node: util for node, util in zip(range(13, 40), util_list)}
        self.game = Fig52Extended()
        self.game.utils = self.utils
        self.nodes = list(range(40))
        self.l = 1 / 40
        self.node_pos = {}
        for i in range(4):
            base = len(self.node_pos)
            row_size = 3 ** i
            for node in [base + j for j in range(row_size)]:
                self.node_pos[node] = ((node - base) / row_size + 1 / (2 * row_size) - self.l / 2,
                                       self.l / 2 + (self.l + (1 - 5 * self.l) / 3) * i)
        self.font("12px Arial")
        self.node_stack = []
        self.explored = {node for node in self.utils}
        self.thick_lines = set()
        self.change_list = []
        self.draw_graph()
        self.stack_manager = self.stack_manager_gen()

    def min_max(self, node):
        game = self.game
        player = game.to_move(node)

        def max_value(node):
            if game.terminal_test(node):
                return game.utility(node, player)
            self.change_list.append(('a', node))
            self.change_list.append(('h',))
            max_a = max(game.actions(node), key=lambda x: min_value(game.result(node, x)))
            max_node = game.result(node, max_a)
            self.utils[node] = self.utils[max_node]
            x1, y1 = self.node_pos[node]
            x2, y2 = self.node_pos[max_node]
            self.change_list.append(('l', (node, max_node - 3 * node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return self.utils[node]

        def min_value(node):
            if game.terminal_test(node):
                return game.utility(node, player)
            self.change_list.append(('a', node))
            self.change_list.append(('h',))
            min_a = min(game.actions(node), key=lambda x: max_value(game.result(node, x)))
            min_node = game.result(node, min_a)
            self.utils[node] = self.utils[min_node]
            x1, y1 = self.node_pos[node]
            x2, y2 = self.node_pos[min_node]
            self.change_list.append(('l', (node, min_node - 3 * node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return self.utils[node]

        return max_value(node)

    def stack_manager_gen(self):
        self.min_max(0)
        for change in self.change_list:
            if change[0] == 'a':
                self.node_stack.append(change[1])
            elif change[0] == 'e':
                self.explored.add(change[1])
            elif change[0] == 'h':
                yield
            elif change[0] == 'l':
                self.thick_lines.add(change[1])
            elif change[0] == 'p':
                self.node_stack.pop()

    def mouse_click(self, x, y):
        try:
            self.stack_manager.send(None)
        except StopIteration:
            pass
        self.draw_graph()

    def draw_graph(self):
        self.clear()
        # draw nodes
        self.stroke(0, 0, 0)
        self.strokeWidth(1)
        # highlight for nodes in stack
        for node in self.node_stack:
            x, y = self.node_pos[node]
            self.fill(200, 200, 0)
            self.rect_n(x - self.l / 5, y - self.l / 5, self.l * 7 / 5, self.l * 7 / 5)
        for node in self.nodes:
            x, y = self.node_pos[node]
            if node in self.explored:
                self.fill(255, 255, 255)
            else:
                self.fill(200, 200, 200)
            self.rect_n(x, y, self.l, self.l)
            self.line_n(x, y, x + self.l, y)
            self.line_n(x, y, x, y + self.l)
            self.line_n(x + self.l, y + self.l, x + self.l, y)
            self.line_n(x + self.l, y + self.l, x, y + self.l)
            self.fill(0, 0, 0)
            if node in self.explored:
                self.text_n(self.utils[node], x + self.l / 10, y + self.l * 9 / 10)
        # draw edges
        for i in range(13):
            x1, y1 = self.node_pos[i][0] + self.l / 2, self.node_pos[i][1] + self.l
            for j in range(3):
                x2, y2 = self.node_pos[i * 3 + j + 1][0] + self.l / 2, self.node_pos[i * 3 + j + 1][1]
                if i in [1, 2, 3]:
                    self.stroke(200, 0, 0)
                else:
                    self.stroke(0, 200, 0)
                if (i, j) in self.thick_lines:
                    self.strokeWidth(3)
                else:
                    self.strokeWidth(1)
                self.line_n(x1, y1, x2, y2)
        self.update()


class Canvas_alpha_beta(Canvas):
    """Alpha-beta pruning for Fig52Extended on HTML canvas"""

    def __init__(self, varname, util_list, width=800, height=600, cid=None):
        super().__init__(varname, width, height, cid)
        self.utils = {node: util for node, util in zip(range(13, 40), util_list)}
        self.game = Fig52Extended()
        self.game.utils = self.utils
        self.nodes = list(range(40))
        self.l = 1 / 40
        self.node_pos = {}
        for i in range(4):
            base = len(self.node_pos)
            row_size = 3 ** i
            for node in [base + j for j in range(row_size)]:
                self.node_pos[node] = ((node - base) / row_size + 1 / (2 * row_size) - self.l / 2,
                                       3 * self.l / 2 + (self.l + (1 - 6 * self.l) / 3) * i)
        self.font("12px Arial")
        self.node_stack = []
        self.explored = {node for node in self.utils}
        self.pruned = set()
        self.ab = {}
        self.thick_lines = set()
        self.change_list = []
        self.draw_graph()
        self.stack_manager = self.stack_manager_gen()

    def alpha_beta_search(self, node):
        game = self.game
        player = game.to_move(node)

        # Functions used by alpha_beta
        def max_value(node, alpha, beta):
            if game.terminal_test(node):
                self.change_list.append(('a', node))
                self.change_list.append(('h',))
                self.change_list.append(('p',))
                return game.utility(node, player)
            v = -np.inf
            self.change_list.append(('a', node))
            self.change_list.append(('ab', node, v, beta))
            self.change_list.append(('h',))
            for a in game.actions(node):
                min_val = min_value(game.result(node, a), alpha, beta)
                if v < min_val:
                    v = min_val
                    max_node = game.result(node, a)
                    self.change_list.append(('ab', node, v, beta))
                if v >= beta:
                    self.change_list.append(('h',))
                    self.pruned.add(node)
                    break
                alpha = max(alpha, v)
            self.utils[node] = v
            if node not in self.pruned:
                self.change_list.append(('l', (node, max_node - 3 * node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return v

        def min_value(node, alpha, beta):
            if game.terminal_test(node):
                self.change_list.append(('a', node))
                self.change_list.append(('h',))
                self.change_list.append(('p',))
                return game.utility(node, player)
            v = np.inf
            self.change_list.append(('a', node))
            self.change_list.append(('ab', node, alpha, v))
            self.change_list.append(('h',))
            for a in game.actions(node):
                max_val = max_value(game.result(node, a), alpha, beta)
                if v > max_val:
                    v = max_val
                    min_node = game.result(node, a)
                    self.change_list.append(('ab', node, alpha, v))
                if v <= alpha:
                    self.change_list.append(('h',))
                    self.pruned.add(node)
                    break
                beta = min(beta, v)
            self.utils[node] = v
            if node not in self.pruned:
                self.change_list.append(('l', (node, min_node - 3 * node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return v

        return max_value(node, -np.inf, np.inf)

    def stack_manager_gen(self):
        self.alpha_beta_search(0)
        for change in self.change_list:
            if change[0] == 'a':
                self.node_stack.append(change[1])
            elif change[0] == 'ab':
                self.ab[change[1]] = change[2:]
            elif change[0] == 'e':
                self.explored.add(change[1])
            elif change[0] == 'h':
                yield
            elif change[0] == 'l':
                self.thick_lines.add(change[1])
            elif change[0] == 'p':
                self.node_stack.pop()

    def mouse_click(self, x, y):
        try:
            self.stack_manager.send(None)
        except StopIteration:
            pass
        self.draw_graph()

    def draw_graph(self):
        self.clear()
        # draw nodes
        self.stroke(0, 0, 0)
        self.strokeWidth(1)
        # highlight for nodes in stack
        for node in self.node_stack:
            x, y = self.node_pos[node]
            # alpha > beta
            if node not in self.explored and self.ab[node][0] > self.ab[node][1]:
                self.fill(200, 100, 100)
            else:
                self.fill(200, 200, 0)
            self.rect_n(x - self.l / 5, y - self.l / 5, self.l * 7 / 5, self.l * 7 / 5)
        for node in self.nodes:
            x, y = self.node_pos[node]
            if node in self.explored:
                if node in self.pruned:
                    self.fill(50, 50, 50)
                else:
                    self.fill(255, 255, 255)
            else:
                self.fill(200, 200, 200)
            self.rect_n(x, y, self.l, self.l)
            self.line_n(x, y, x + self.l, y)
            self.line_n(x, y, x, y + self.l)
            self.line_n(x + self.l, y + self.l, x + self.l, y)
            self.line_n(x + self.l, y + self.l, x, y + self.l)
            self.fill(0, 0, 0)
            if node in self.explored and node not in self.pruned:
                self.text_n(self.utils[node], x + self.l / 10, y + self.l * 9 / 10)
        # draw edges
        for i in range(13):
            x1, y1 = self.node_pos[i][0] + self.l / 2, self.node_pos[i][1] + self.l
            for j in range(3):
                x2, y2 = self.node_pos[i * 3 + j + 1][0] + self.l / 2, self.node_pos[i * 3 + j + 1][1]
                if i in [1, 2, 3]:
                    self.stroke(200, 0, 0)
                else:
                    self.stroke(0, 200, 0)
                if (i, j) in self.thick_lines:
                    self.strokeWidth(3)
                else:
                    self.strokeWidth(1)
                self.line_n(x1, y1, x2, y2)
        # display alpha and beta
        for node in self.node_stack:
            if node not in self.explored:
                x, y = self.node_pos[node]
                alpha, beta = self.ab[node]
                self.text_n(alpha, x - self.l / 2, y - self.l / 10)
                self.text_n(beta, x + self.l, y - self.l / 10)
        self.update()


class Canvas_fol_bc_ask(Canvas):
    """fol_bc_ask() on HTML canvas"""

    def __init__(self, varname, kb, query, width=800, height=600, cid=None):
        super().__init__(varname, width, height, cid)
        self.kb = kb
        self.query = query
        self.l = 1 / 20
        self.b = 3 * self.l
        bc_out = list(self.fol_bc_ask())
        if len(bc_out) == 0:
            self.valid = False
        else:
            self.valid = True
            graph = bc_out[0][0][0]
            s = bc_out[0][1]
            while True:
                new_graph = subst(s, graph)
                if graph == new_graph:
                    break
                graph = new_graph
            self.make_table(graph)
        self.context = None
        self.draw_table()

    def fol_bc_ask(self):
        KB = self.kb
        query = self.query

        def fol_bc_or(KB, goal, theta):
            for rule in KB.fetch_rules_for_goal(goal):
                lhs, rhs = parse_definite_clause(standardize_variables(rule))
                for theta1 in fol_bc_and(KB, lhs, unify_mm(rhs, goal, theta)):
                    yield ([(goal, theta1[0])], theta1[1])

        def fol_bc_and(KB, goals, theta):
            if theta is None:
                pass
            elif not goals:
                yield ([], theta)
            else:
                first, rest = goals[0], goals[1:]
                for theta1 in fol_bc_or(KB, subst(theta, first), theta):
                    for theta2 in fol_bc_and(KB, rest, theta1[1]):
                        yield (theta1[0] + theta2[0], theta2[1])

        return fol_bc_or(KB, query, {})

    def make_table(self, graph):
        table = []
        pos = {}
        links = set()
        edges = set()

        def dfs(node, depth):
            if len(table) <= depth:
                table.append([])
            pos = len(table[depth])
            table[depth].append(node[0])
            for child in node[1]:
                child_id = dfs(child, depth + 1)
                links.add(((depth, pos), child_id))
            return (depth, pos)

        dfs(graph, 0)
        y_off = 0.85 / len(table)
        for i, row in enumerate(table):
            x_off = 0.95 / len(row)
            for j, node in enumerate(row):
                pos[(i, j)] = (0.025 + j * x_off + (x_off - self.b) / 2, 0.025 + i * y_off + (y_off - self.l) / 2)
        for p, c in links:
            x1, y1 = pos[p]
            x2, y2 = pos[c]
            edges.add((x1 + self.b / 2, y1 + self.l, x2 + self.b / 2, y2))

        self.table = table
        self.pos = pos
        self.edges = edges

    def mouse_click(self, x, y):
        x, y = x / self.width, y / self.height
        for node in self.pos:
            xs, ys = self.pos[node]
            xe, ye = xs + self.b, ys + self.l
            if xs <= x <= xe and ys <= y <= ye:
                self.context = node
                break
        self.draw_table()

    def draw_table(self):
        self.clear()
        self.strokeWidth(3)
        self.stroke(0, 0, 0)
        self.font("12px Arial")
        if self.valid:
            # draw nodes
            for i, j in self.pos:
                x, y = self.pos[(i, j)]
                self.fill(200, 200, 200)
                self.rect_n(x, y, self.b, self.l)
                self.line_n(x, y, x + self.b, y)
                self.line_n(x, y, x, y + self.l)
                self.line_n(x + self.b, y, x + self.b, y + self.l)
                self.line_n(x, y + self.l, x + self.b, y + self.l)
                self.fill(0, 0, 0)
                self.text_n(self.table[i][j], x + 0.01, y + self.l - 0.01)
            # draw edges
            for x1, y1, x2, y2 in self.edges:
                self.line_n(x1, y1, x2, y2)
        else:
            self.fill(255, 0, 0)
            self.rect_n(0, 0, 1, 1)
        # text area
        self.fill(255, 255, 255)
        self.rect_n(0, 0.9, 1, 0.1)
        self.strokeWidth(5)
        self.stroke(0, 0, 0)
        self.line_n(0, 0.9, 1, 0.9)
        self.font("22px Arial")
        self.fill(0, 0, 0)
        self.text_n(self.table[self.context[0]][self.context[1]] if self.context else "Click for text", 0.025, 0.975)
        self.update()


############################################################################################################

#####################           Functions to assist plotting in search.ipynb            ####################

############################################################################################################


def show_map(graph_data, node_colors=None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights = graph_data['edge_weights']

    # set the size of the plot
    plt.figure(figsize=(18, 13))
    # draw the graph (both nodes and edges) with locations from romania_locations
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()},
            node_color=[node_colors[node] for node in G.nodes()], linewidths=0.3, edgecolors='k')

    # draw labels for nodes
    node_label_handles = nx.draw_networkx_labels(G, pos=node_label_pos, font_size=14)

    # add a white bounding box behind the node labels
    [label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in node_label_handles.values()]

    # add edge lables to the graph
    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=14)

    # add a legend
    white_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="white")
    orange_circle = lines.Line2D([], [], color="orange", marker='o', markersize=15, markerfacecolor="orange")
    red_circle = lines.Line2D([], [], color="red", marker='o', markersize=15, markerfacecolor="red")
    gray_circle = lines.Line2D([], [], color="gray", marker='o', markersize=15, markerfacecolor="gray")
    green_circle = lines.Line2D([], [], color="green", marker='o', markersize=15, markerfacecolor="green")
    plt.legend((white_circle, orange_circle, red_circle, gray_circle, green_circle),
               ('Un-explored', 'Frontier', 'Currently Exploring', 'Explored', 'Final Solution'),
               numpoints=1, prop={'size': 16}, loc=(.8, .75))

    # show the plot. No need to use in notebooks. nx.draw will show the graph itself.
    plt.show()


# helper functions for visualisations

def final_path_colors(initial_node_colors, problem, solution):
    "Return a node_colors dict of the final path provided the problem and solution."

    # get initial node colors
    final_colors = dict(initial_node_colors)
    # color all the nodes in solution and starting node to green
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors


def display_visual(graph_data, user_input, algorithm=None, problem=None):
    initial_node_colors = graph_data['node_colors']
    if user_input is False:
        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                global all_node_colors

                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i
                    # time.sleep(.5)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

    if user_input is True:
        node_colors = dict(initial_node_colors)
        if isinstance(algorithm, dict):
            assert set(algorithm.keys()).issubset({"Breadth First Tree Search",
                                                   "Depth First Tree Search",
                                                   "Breadth First Search",
                                                   "Depth First Graph Search",
                                                   "Best First Graph Search",
                                                   "Uniform Cost Search",
                                                   "Depth Limited Search",
                                                   "Iterative Deepening Search",
                                                   "Greedy Best First Search",
                                                   "A-star Search",
                                                   "Recursive Best First Search"})

            algo_dropdown = widgets.Dropdown(description="Search algorithm: ",
                                             options=sorted(list(algorithm.keys())),
                                             value="Breadth First Tree Search")
            display(algo_dropdown)
        elif algorithm is None:
            print("No algorithm to run.")
            return 0

        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                problem = GraphProblem(start_dropdown.value, end_dropdown.value, romania_map)
                global all_node_colors

                user_algorithm = algorithm[algo_dropdown.value]

                iterations, all_node_colors, node = user_algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i
                    # time.sleep(.5)

        start_dropdown = widgets.Dropdown(description="Start city: ",
                                          options=sorted(list(node_colors.keys())), value="Arad")
        display(start_dropdown)

        end_dropdown = widgets.Dropdown(description="Goal city: ",
                                        options=sorted(list(node_colors.keys())), value="Fagaras")
        display(end_dropdown)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)


# Function to plot NQueensCSP in csp.py and NQueensProblem in search.py
def plot_NQueens(solution):
    n = len(solution)
    board = np.array([2 * int((i + j) % 2) for j in range(n) for i in range(n)]).reshape((n, n))
    im = Image.open('images/queen_s.png')
    height = im.size[1]
    im = np.array(im).astype(np.float) / 255
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('{} Queens'.format(n))
    plt.imshow(board, cmap='binary', interpolation='nearest')
    # NQueensCSP gives a solution as a dictionary
    if isinstance(solution, dict):
        for (k, v) in solution.items():
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    # NQueensProblem gives a solution as a list
    elif isinstance(solution, list):
        for (k, v) in enumerate(solution):
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    fig.tight_layout()
    plt.show()


# Function to plot a heatmap, given a grid
def heatmap(grid, cmap='binary', interpolation='nearest'):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('Heatmap')
    plt.imshow(grid, cmap=cmap, interpolation=interpolation)
    fig.tight_layout()
    plt.show()


# Generates a gaussian kernel
def gaussian_kernel(l=5, sig=1.0):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel


# Plots utility function for a POMDP
def plot_pomdp_utility(utility):
    save = utility['0'][0]
    delete = utility['1'][0]
    ask_save = utility['2'][0]
    ask_delete = utility['2'][-1]
    left = (save[0] - ask_save[0]) / (save[0] - ask_save[0] + ask_save[1] - save[1])
    right = (delete[0] - ask_delete[0]) / (delete[0] - ask_delete[0] + ask_delete[1] - delete[1])

    colors = ['g', 'b', 'k']
    for action in utility:
        for value in utility[action]:
            plt.plot(value, color=colors[int(action)])
    plt.vlines([left, right], -20, 10, linestyles='dashed', colors='c')
    plt.ylim(-20, 13)
    plt.xlim(0, 1)
    plt.text(left / 2 - 0.05, 10, 'Save')
    plt.text((right + left) / 2 - 0.02, 10, 'Ask')
    plt.text((right + 1) / 2 - 0.07, 10, 'Delete')
    plt.show()
