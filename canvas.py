from IPython.display import HTML, display
from utils import argmax, argmin
from games import TicTacToe, alphabeta_player, random_player, Fig52Extended, infinity

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
    IPython must be able to refernce the variable name that is being passed.
    """

    def __init__(self, varname, width=800, height=600, cid=None):
        """"""
        self.name = varname
        self.cid = cid or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.cid, self.width, self.height, self.name)
        self.exec_list = []
        display_html(self.html)

    def mouse_click(self, x, y):
        "Override this method to handle mouse click at position (x, y)"
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def execute(self, exec_str):
        "Stores the command to be exectued to a list which is used later during update()"
        if not isinstance(exec_str, str):
            print("Invalid execution argument:", exec_str)
            self.alert("Recieved invalid execution command format")
        prefix = "{0}_canvas_object.".format(self.cid)
        self.exec_list.append(prefix + exec_str + ';')

    def fill(self, r, g, b):
        "Changes the fill color to a color in rgb format"
        self.execute("fill({0}, {1}, {2})".format(r, g, b))

    def stroke(self, r, g, b):
        "Changes the colors of line/strokes to rgb"
        self.execute("stroke({0}, {1}, {2})".format(r, g, b))

    def strokeWidth(self, w):
        "Changes the width of lines/strokes to 'w' pixels"
        self.execute("strokeWidth({0})".format(w))

    def rect(self, x, y, w, h):
        "Draw a rectangle with 'w' width, 'h' height and (x, y) as the top-left corner"
        self.execute("rect({0}, {1}, {2}, {3})".format(x, y, w, h))

    def rect_n(self, xn, yn, wn, hn):
        "Similar to rect(), but the dimensions are normalized to fall between 0 and 1"
        x = round(xn * self.width)
        y = round(yn * self.height)
        w = round(wn * self.width)
        h = round(hn * self.height)
        self.rect(x, y, w, h)

    def line(self, x1, y1, x2, y2):
        "Draw a line from (x1, y1) to (x2, y2)"
        self.execute("line({0}, {1}, {2}, {3})".format(x1, y1, x2, y2))

    def line_n(self, x1n, y1n, x2n, y2n):
        "Similar to line(), but the dimensions are normalized to fall between 0 and 1"
        x1 = round(x1n * self.width)
        y1 = round(y1n * self.height)
        x2 = round(x2n * self.width)
        y2 = round(y2n * self.height)
        self.line(x1, y1, x2, y2)

    def arc(self, x, y, r, start, stop):
        "Draw an arc with (x, y) as centre, 'r' as radius from angles 'start' to 'stop'"
        self.execute("arc({0}, {1}, {2}, {3}, {4})".format(x, y, r, start, stop))

    def arc_n(self, xn, yn, rn, start, stop):
        """Similar to arc(), but the dimensions are normalized to fall between 0 and 1
        The normalizing factor for radius is selected between width and height by
        seeing which is smaller
        """
        x = round(xn * self.width)
        y = round(yn * self.height)
        r = round(rn * min(self.width, self.height))
        self.arc(x, y, r, start, stop)

    def clear(self):
        "Clear the HTML canvas"
        self.execute("clear()")

    def font(self, font):
        "Changes the font of text"
        self.execute('font("{0}")'.format(font))

    def text(self, txt, x, y, fill=True):
        "Display a text at (x, y)"
        if fill:
            self.execute('fill_text("{0}", {1}, {2})'.format(txt, x, y))
        else:
            self.execute('stroke_text("{0}", {1}, {2})'.format(txt, x, y))

    def text_n(self, txt, xn, yn, fill=True):
        "Similar to text(), but with normalized coordinates"
        x = round(xn * self.width)
        y = round(yn * self.height)
        self.text(txt, x, y, fill)

    def alert(self, message):
        "Immediately display an alert"
        display_html('<script>alert("{0}")</script>'.format(message))

    def update(self):
        "Execute the JS code to execute the commands queued by execute()"
        exec_code = "<script>\n" + '\n'.join(self.exec_list) + "\n</script>"
        self.exec_list = []
        display_html(exec_code)


def display_html(html_string):
    display(HTML(html_string))


################################################################################
    

class Canvas_TicTacToe(Canvas):
    """Play a 3x3 TicTacToe game on HTML canvas
    """
    def __init__(self, varname, player_1='human', player_2='random',
                 width=300, height=350, cid=None):
        valid_players = ('human', 'random', 'alphabeta')
        if player_1 not in valid_players or player_2 not in valid_players:
            raise TypeError("Players must be one of {}".format(valid_players))
        Canvas.__init__(self, varname, width, height, cid)
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
            if 0.55 <= x/self.width <= 0.95 and 6/7 <= y/self.height <= 6/7+1/8:
                self.state = self.ttt.initial
                self.turn = 0
                self.draw_board()
            return

        if player == 'human':
            x, y = int(3*x/self.width) + 1, int(3*y/(self.height*6/7)) + 1
            if (x, y) not in self.ttt.actions(self.state):
                # Invalid move
                return
            move = (x, y)
        elif player == 'alphabeta':
            move = alphabeta_player(self.ttt, self.state)
        else:
            move = random_player(self.ttt, self.state)
        self.state = self.ttt.result(self.state, move)
        self.turn ^= 1
        self.draw_board()

    def draw_board(self):
        self.clear()
        self.stroke(0, 0, 0)
        offset = 1/20
        self.line_n(0 + offset, (1/3)*6/7, 1 - offset, (1/3)*6/7)
        self.line_n(0 + offset, (2/3)*6/7, 1 - offset, (2/3)*6/7)
        self.line_n(1/3, (0 + offset)*6/7, 1/3, (1 - offset)*6/7)
        self.line_n(2/3, (0 + offset)*6/7, 2/3, (1 - offset)*6/7)

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
                self.text_n('Game Draw!', offset, 6/7 + offset)
            else:
                self.text_n('Player {} wins!'.format("XO"[utility < 0]), offset, 6/7 + offset)
                # Find the 3 and draw a line
                self.stroke([255, 0][self.turn], [0, 255][self.turn], 0)
                for i in range(3):
                    if all([(i + 1, j + 1) in self.state.board for j in range(3)]) and \
                       len({self.state.board[(i + 1, j + 1)] for j in range(3)}) == 1:
                        self.line_n(i/3 + 1/6, offset*6/7, i/3 + 1/6, (1 - offset)*6/7)
                    if all([(j + 1, i + 1) in self.state.board for j in range(3)]) and \
                       len({self.state.board[(j + 1, i + 1)] for j in range(3)}) == 1:
                        self.line_n(offset, (i/3 + 1/6)*6/7, 1 - offset, (i/3 + 1/6)*6/7)
                if all([(i + 1, i + 1) in self.state.board for i in range(3)]) and \
                   len({self.state.board[(i + 1, i + 1)] for i in range(3)}) == 1:
                        self.line_n(offset, offset*6/7, 1 - offset, (1 - offset)*6/7)
                if all([(i + 1, 3 - i) in self.state.board for i in range(3)]) and \
                   len({self.state.board[(i + 1, 3 - i)] for i in range(3)}) == 1:
                        self.line_n(offset, (1 - offset)*6/7, 1 - offset, offset*6/7)
            # restart button
            self.fill(0, 0, 255)
            self.rect_n(0.5 + offset, 6/7, 0.4, 1/8)
            self.fill(0, 0, 0)
            self.text_n('Restart', 0.5 + 2*offset, 13/14)
        else:  # Print which player's turn it is
            self.text_n("Player {}'s move({})".format("XO"[self.turn], self.players[self.turn]),
                        offset, 6/7 + offset)

        self.update()

    def draw_x(self, position):
        self.stroke(0, 255, 0)
        x, y = [i-1 for i in position]
        offset = 1/15
        self.line_n(x/3 + offset, (y/3 + offset)*6/7, x/3 + 1/3 - offset, (y/3 + 1/3 - offset)*6/7)
        self.line_n(x/3 + 1/3 - offset, (y/3 + offset)*6/7, x/3 + offset, (y/3 + 1/3 - offset)*6/7)

    def draw_o(self, position):
        self.stroke(255, 0, 0)
        x, y = [i-1 for i in position]
        self.arc_n(x/3 + 1/6, (y/3 + 1/6)*6/7, 1/9, 0, 360)


class Canvas_minimax(Canvas):
    """Minimax for Fig52Extended on HTML canvas
    """
    def __init__(self, varname, util_list, width=800, height=600, cid=None):
        Canvas.__init__(self, varname, width, height, cid)
        self.utils = {node:util for node, util in zip(range(13, 40), util_list)}
        self.game = Fig52Extended()
        self.game.utils = self.utils
        self.nodes = list(range(40))
        self.l = 1/40
        self.node_pos = {}
        for i in range(4):
            base = len(self.node_pos)
            row_size = 3**i
            for node in [base + j for j in range(row_size)]:
                self.node_pos[node] = ((node - base)/row_size + 1/(2*row_size) - self.l/2,
                                       self.l/2 + (self.l + (1 - 5*self.l)/3)*i)
        self.font("12px Arial")
        self.node_stack = []
        self.explored = {node for node in self.utils}
        self.thick_lines = set()
        self.change_list = []
        self.draw_graph()
        self.stack_manager = self.stack_manager_gen()

    def minimax(self, node):
        game = self.game
        player = game.to_move(node)
        def max_value(node):
            if game.terminal_test(node):
                return game.utility(node, player)
            self.change_list.append(('a', node))
            self.change_list.append(('h',))
            max_a = argmax(game.actions(node), key=lambda x: min_value(game.result(node, x)))
            max_node = game.result(node, max_a)
            self.utils[node] = self.utils[max_node]
            x1, y1 = self.node_pos[node]
            x2, y2 = self.node_pos[max_node]
            self.change_list.append(('l', (node, max_node - 3*node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return self.utils[node]

        def min_value(node):
            if game.terminal_test(node):
                return game.utility(node, player)
            self.change_list.append(('a', node))
            self.change_list.append(('h',))
            min_a = argmin(game.actions(node), key=lambda x: max_value(game.result(node, x)))
            min_node = game.result(node, min_a)
            self.utils[node] = self.utils[min_node]
            x1, y1 = self.node_pos[node]
            x2, y2 = self.node_pos[min_node]
            self.change_list.append(('l', (node, min_node - 3*node - 1)))
            self.change_list.append(('e', node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return self.utils[node]

        return max_value(node)

    def stack_manager_gen(self):
        self.minimax(0)
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
            self.rect_n(x - self.l/5, y - self.l/5, self.l*7/5, self.l*7/5)
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
                self.text_n(self.utils[node], x + self.l/10, y + self.l*9/10)
        # draw edges
        for i in range(13):
            x1, y1 = self.node_pos[i][0] + self.l/2, self.node_pos[i][1] + self.l
            for j in range(3):
                x2, y2 = self.node_pos[i*3 + j + 1][0] + self.l/2, self.node_pos[i*3 + j + 1][1]
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


class Canvas_alphabeta(Canvas):
    """Alpha-beta pruning for Fig52Extended on HTML canvas
    """
    def __init__(self, varname, util_list, width=800, height=600, cid=None):
        Canvas.__init__(self, varname, width, height, cid)
        self.utils = {node:util for node, util in zip(range(13, 40), util_list)}
        self.game = Fig52Extended()
        self.game.utils = self.utils
        self.nodes = list(range(40))
        self.l = 1/40
        self.node_pos = {}
        for i in range(4):
            base = len(self.node_pos)
            row_size = 3**i
            for node in [base + j for j in range(row_size)]:
                self.node_pos[node] = ((node - base)/row_size + 1/(2*row_size) - self.l/2,
                                       3*self.l/2 + (self.l + (1 - 6*self.l)/3)*i)
        self.font("12px Arial")
        self.node_stack = []
        self.explored = {node for node in self.utils}
        self.pruned = set()
        self.ab = {}
        self.thick_lines = set()
        self.change_list = []
        self.draw_graph()
        self.stack_manager = self.stack_manager_gen()

    def alphabeta_search(self, node):
        game = self.game
        player = game.to_move(node)

        # Functions used by alphabeta
        def max_value(node, alpha, beta):
            if game.terminal_test(node):
                self.change_list.append(('a', node))
                self.change_list.append(('h',))
                self.change_list.append(('p',))
                return game.utility(node, player)
            v = -infinity
            self.change_list.append(('a', node))
            self.change_list.append(('ab',node, v, beta))
            self.change_list.append(('h',))
            for a in game.actions(node):
                min_val = min_value(game.result(node, a), alpha, beta)
                if v < min_val:
                    v = min_val
                    max_node = game.result(node, a)
                    self.change_list.append(('ab',node, v, beta))
                if v >= beta:
                    self.change_list.append(('h',))
                    self.pruned.add(node)
                    break
                alpha = max(alpha, v)
            self.utils[node] = v
            if node not in self.pruned:
                self.change_list.append(('l', (node, max_node - 3*node - 1)))
            self.change_list.append(('e',node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return v

        def min_value(node, alpha, beta):
            if game.terminal_test(node):
                self.change_list.append(('a', node))
                self.change_list.append(('h',))
                self.change_list.append(('p',))
                return game.utility(node, player)
            v = infinity
            self.change_list.append(('a', node))
            self.change_list.append(('ab',node, alpha, v))
            self.change_list.append(('h',))
            for a in game.actions(node):
                max_val = max_value(game.result(node, a), alpha, beta)
                if v > max_val:
                    v = max_val
                    min_node = game.result(node, a)
                    self.change_list.append(('ab',node, alpha, v))
                if v <= alpha:
                    self.change_list.append(('h',))
                    self.pruned.add(node)
                    break
                beta = min(beta, v)
            self.utils[node] = v
            if node not in self.pruned:
                self.change_list.append(('l', (node, min_node - 3*node - 1)))
            self.change_list.append(('e',node))
            self.change_list.append(('p',))
            self.change_list.append(('h',))
            return v

        return max_value(node, -infinity, infinity)

    def stack_manager_gen(self):
        self.alphabeta_search(0)
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
            self.rect_n(x - self.l/5, y - self.l/5, self.l*7/5, self.l*7/5)
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
                self.text_n(self.utils[node], x + self.l/10, y + self.l*9/10)
        # draw edges
        for i in range(13):
            x1, y1 = self.node_pos[i][0] + self.l/2, self.node_pos[i][1] + self.l
            for j in range(3):
                x2, y2 = self.node_pos[i*3 + j + 1][0] + self.l/2, self.node_pos[i*3 + j + 1][1]
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
                self.text_n(alpha, x - self.l/2, y - self.l/10)
                self.text_n(beta, x + self.l, y - self.l/10)
        self.update()
