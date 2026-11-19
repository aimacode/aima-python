"""
Interactive views of the algorithms, built on ipywidgets alone.

The views are drawn as SVG inside ipywidgets.HTML, and the animations are timed
by ipywidgets.Play in the browser, so they run in classic Jupyter, in JupyterLab
and in JupyterLite (Pyodide), with no server and no other widget library.

A Player replays a list of frames, recorded beforehand; the search functions are
recorded with RecordingProblem (in aima.search), which wraps a problem and leaves
the functions themselves unchanged. The ready-made players are:

    graph_search_player   a search on a graph with locations, e.g. the Romania map
    eight_puzzle_player   the solution path of a search on the 8-puzzle
    n_queens_player       a search on NQueensProblem, one board per expansion

and EightPuzzleGame lets the user solve an 8-puzzle by clicking the tiles.
"""

import itertools
import random
from html import escape

import ipywidgets

from aima.search import (EightPuzzle, Node, RecordingProblem, astar_search)

# colors of the views (state of a node of the search, and the board squares)
COLORS = dict(unreached='#ffffff', frontier='#f6c453', explored='#a9c4e8',
              current='#d9534f', path='#2a9d8f', edge='#b8b8b8', text='#222222',
              light='#f0d9b5', dark='#b58863', tile='#e8eef7', attacked='#d9534f')

_ids = itertools.count()


def svg_page(body, width, height, max_width):
    """Wrap SVG elements in an <svg> scaled to the output width (up to max_width
    pixels), on a white background so that it reads in dark themes too."""
    return (f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{max_width}px;'
            f'height:auto;font-family:sans-serif" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>{body}</svg>')


# ______________________________________________________________________________
# Step-through player


class Player(ipywidgets.VBox):
    """Replay a list of frames with Play/Pause, Step, Reset, a slider over the
    frames and a speed control. render(player, i) returns the HTML of frame
    player.frames[i] (player.shown is the index on show before it, or None); the
    default speed is one frame per second."""

    def __init__(self, frames, render, speed=1.0):
        self.frames = frames
        self.render = render
        last = max(len(frames) - 1, 0)
        self.timer = ipywidgets.Play(min=0, max=last, interval=int(1000 / speed),
                                     show_repeat=False, layout=ipywidgets.Layout(display='none'))
        self.slider = ipywidgets.IntSlider(min=0, max=last, description='Step',
                                           continuous_update=True,
                                           layout=ipywidgets.Layout(width='360px'))
        ipywidgets.jslink((self.timer, 'value'), (self.slider, 'value'))
        self.play_button = ipywidgets.ToggleButton(description='Play', icon='play',
                                                   layout=ipywidgets.Layout(width='90px'))
        ipywidgets.jslink((self.play_button, 'value'), (self.timer, 'playing'))
        self.step_button = ipywidgets.Button(description='Step', icon='step-forward',
                                             layout=ipywidgets.Layout(width='90px'))
        self.reset_button = ipywidgets.Button(description='Reset', icon='undo',
                                              layout=ipywidgets.Layout(width='90px'))
        self.speed = ipywidgets.FloatSlider(value=speed, min=0.25, max=4, step=0.25,
                                            description='Steps/s', readout_format='.2f',
                                            layout=ipywidgets.Layout(width='280px'))
        self.view = ipywidgets.HTML()
        self.shown = None
        self.slider.observe(lambda change: self.show(change['new']), 'value')
        self.play_button.observe(self.on_play, 'value')
        self.step_button.on_click(lambda button: self.step())
        self.reset_button.on_click(lambda button: self.reset())
        self.speed.observe(self.on_speed, 'value')
        controls = ipywidgets.HBox([self.play_button, self.step_button, self.reset_button,
                                    self.slider, self.speed, self.timer],
                                   layout=ipywidgets.Layout(flex_flow='row wrap'))
        super().__init__([controls, self.view])
        self.add_class('aima-player')
        self.show(0)

    @property
    def value(self):
        """The index of the frame on show."""
        return self.slider.value

    def show(self, i):
        """Draw frame i."""
        self.view.value = self.render(self, i)
        self.shown = i
        if i == len(self.frames) - 1:
            self.play_button.value = False

    def on_speed(self, change):
        """Set the time between frames from the speed, in frames per second."""
        self.timer.interval = int(1000 / change['new'])

    def on_play(self, change):
        """Restart from the first frame when Play is pressed on the last one."""
        self.play_button.description = 'Pause' if change['new'] else 'Play'
        self.play_button.icon = 'pause' if change['new'] else 'play'
        if change['new'] and self.slider.value == self.slider.max:
            self.slider.value = 0

    def step(self):
        """Pause and move one frame forward."""
        self.play_button.value = False
        if self.slider.value < self.slider.max:
            self.slider.value += 1

    def reset(self):
        """Pause and go back to the first frame."""
        self.play_button.value = False
        self.slider.value = 0


# ______________________________________________________________________________
# Search on a graph with locations (the Romania map)


def graph_search_svg(graph, step, initial, goal):
    """Draw a graph whose nodes have graph.locations, colored by a step of
    RecordingProblem.steps(): frontier, explored, the node being expanded and, in
    the last step, the path found. Return the SVG as a string."""
    xs = [x for x, y in graph.locations.values()]
    ys = [y for x, y in graph.locations.values()]
    left, top, scale, pad = min(xs), max(ys), 1.0, 40
    width, height = int(max(xs) - left) + 2 * pad + 60, int(top - min(ys)) + 2 * pad

    def at(node):
        x, y = graph.locations[node]
        return (x - left) * scale + pad, (top - y) * scale + pad

    path = step.get('path') or []
    on_path = set(zip(path, path[1:])) | set(zip(path[1:], path))
    body = []
    drawn = set()
    for a in graph.nodes():
        for b, d in graph.get(a).items():
            if (b, a) in drawn:
                continue
            drawn.add((a, b))
            (x1, y1), (x2, y2) = at(a), at(b)
            color, width_ = (COLORS['path'], 6) if (a, b) in on_path else (COLORS['edge'], 2)
            body.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="{color}" stroke-width="{width_}"/>')
            body.append(f'<text x="{(x1 + x2) / 2:.0f}" y="{(y1 + y2) / 2 + 4:.0f}" '
                        f'font-size="11" fill="#666666" stroke="#ffffff" stroke-width="3" '
                        f'paint-order="stroke" text-anchor="middle">{d}</text>')
    for node in graph.nodes():
        x, y = at(node)
        if node == step['current']:
            fill = COLORS['current']
        elif node in step['explored']:
            fill = COLORS['explored']
        elif node in step['frontier']:
            fill = COLORS['frontier']
        else:
            fill = COLORS['unreached']
        ring, ring_width = (COLORS['path'], 3) if node in path else ('#555555', 1.5)
        body.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="9" fill="{fill}" stroke="{ring}" '
                    f'stroke-width="{ring_width}"/>')
        weight = 'bold' if node in (initial, goal) else 'normal'
        body.append(f'<text x="{x + 12:.0f}" y="{y - 8:.0f}" font-size="13" font-weight="{weight}" '
                    f'fill="{COLORS["text"]}" stroke="#ffffff" stroke-width="3" '
                    f'paint-order="stroke">{escape(str(node))}</text>')
    legend = [('frontier', 'frontier'), ('explored', 'explored'), ('current', 'expanding'),
              ('unreached', 'not reached')]
    for i, (key, label) in enumerate(legend):
        x = pad + 110 * i
        body.append(f'<circle cx="{x}" cy="{height - 14}" r="7" fill="{COLORS[key]}" '
                    f'stroke="#555555"/><text x="{x + 12}" y="{height - 10}" font-size="12" '
                    f'fill="{COLORS["text"]}">{label}</text>')
    return svg_page(''.join(body), width, height, 680)


def graph_search_caption(steps, i, goal):
    """Describe step i of a recorded graph search in one or two lines of HTML."""
    step = steps[i]
    if 'path' in step:
        path = ' &rarr; '.join(escape(str(state)) for state in step['path'])
        return (f'<b>Goal reached</b> after {len(step["explored"])} expansions: {path} '
                f'(cost {step["cost"]}).')
    frontier = escape(', '.join(sorted(map(str, step['frontier']))) or 'empty')
    if step['current'] is None:
        return f'Step 0 of {len(steps) - 1}: the frontier holds the initial state {frontier}.'
    text = (f'Step {i} of {len(steps) - 1}: expand <b>{escape(str(step["current"]))}</b>. '
            f'Frontier: {frontier}. Explored: {len(step["explored"])} states.')
    if i == len(steps) - 1 and goal is None:
        text += ' No solution.'
    return text


def graph_search_player(search, problem, *args, speed=1.0, **kwargs):
    """Run ``search(problem, *args, **kwargs)`` on a GraphProblem whose graph has
    locations (e.g. romania_map) and return a Player that replays it, one
    expansion per step, ending with the path found and its cost."""
    recorder = RecordingProblem(problem)
    goal = search(recorder, *args, **kwargs)
    steps = recorder.steps(goal)

    def render(player, i):
        svg = graph_search_svg(problem.graph, steps[i], problem.initial, problem.goal)
        return f'{svg}<div style="max-width:680px">{graph_search_caption(steps, i, goal)}</div>'

    return Player(steps, render, speed)


# ______________________________________________________________________________
# 8-puzzle


def eight_puzzle_svg(state, previous=None, duration=0.4, size=3):
    """Draw a sliding-tile board (0 is the blank). If the previous state is given
    and differs by one move, the moved tile slides from its old square."""
    cell, gap = 70, 4
    side = size * cell + gap
    key = f'aima-slide-{next(_ids)}'
    moved, style = None, ''
    if previous is not None and previous != state:
        blank, old_blank = state.index(0), previous.index(0)
        moved = old_blank  # the moved tile now sits where the blank was
        dx = (blank % size - old_blank % size) * cell
        dy = (blank // size - old_blank // size) * cell
        style = (f'<style>@keyframes {key} {{ from {{ transform: translate({dx}px,{dy}px); }} '
                 f'to {{ transform: translate(0,0); }} }} '
                 f'.{key} {{ animation: {key} {duration}s ease-in-out; }}</style>')
    body = [style, f'<rect width="{side}" height="{side}" fill="#555555" rx="6"/>']
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        x, y, w = (i % size) * cell + gap, (i // size) * cell + gap, cell - gap
        body.append(f'<g class="{key if i == moved else ""}">'
                    f'<rect x="{x}" y="{y}" width="{w}" height="{w}" rx="6" '
                    f'fill="{COLORS["tile"]}"/><text x="{x + w / 2:.0f}" y="{y + w / 2 + 10:.0f}" '
                    f'font-size="30" text-anchor="middle" '
                    f'fill="{COLORS["text"]}">{tile}</text></g>')
    return svg_page(''.join(body), side, side, side)


def eight_puzzle_player(search, problem, *args, speed=1.0, **kwargs):
    """Run ``search(problem, *args, **kwargs)`` on an EightPuzzle and return a Player
    that replays the solution found, one move per step, with the tiles sliding."""
    recorder = RecordingProblem(problem)
    goal = search(recorder, *args, **kwargs)
    states = goal.path_states() if goal else [problem.initial]

    def render(player, i):
        # slide the tile only between consecutive moves, not on a jump of the slider
        previous = None
        if player.shown is not None and abs(player.shown - i) == 1:
            previous = states[player.shown]
        if goal is None:
            caption = f'No solution found ({recorder.succs} nodes expanded).'
        else:
            caption = (f'Move {i} of {len(states) - 1}; the search expanded {recorder.succs} nodes '
                       f'to find this solution.')
        svg = eight_puzzle_svg(states[i], previous, duration=min(0.4, 0.6 / player.speed.value))
        return f'{svg}<div>{caption}</div>'

    return Player(states, render, speed)


def manhattan(node, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0), size=3):
    """Sum of the distances of the tiles from their goal squares (an admissible
    heuristic for the sliding-tile puzzles)."""
    state = node.state if isinstance(node, Node) else node
    where = {tile: i for i, tile in enumerate(goal)}
    return sum(abs(i // size - where[t] // size) + abs(i % size - where[t] % size)
               for i, t in enumerate(state) if t != 0)


def scrambled_eight_puzzle(moves=20, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0), seed=None):
    """Return a state reached from the goal by random moves of the blank (so it is
    always solvable), never undoing the previous move."""
    rng = random.Random(seed)
    problem = EightPuzzle(goal, goal)
    opposite = dict(UP='DOWN', DOWN='UP', LEFT='RIGHT', RIGHT='LEFT')
    state, last = goal, None
    for _ in range(moves):
        action = rng.choice([a for a in problem.actions(state) if a != opposite.get(last)])
        state, last = problem.result(state, action), action
    return state


def moves(n):
    """'1 move', '2 moves', ..."""
    return f'{n} move{"" if n == 1 else "s"}'


class EightPuzzleGame(ipywidgets.VBox):
    """An 8-puzzle the user solves by clicking a tile next to the blank. Any other
    click is refused with a message; when the puzzle is solved the number of
    moves is compared with the optimum found by A* (Manhattan heuristic)."""

    def __init__(self, initial=None, scramble=16, seed=None):
        self.goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.scramble, self.rng = scramble, random.Random(seed)
        self.buttons = [ipywidgets.Button(layout=ipywidgets.Layout(width='64px', height='64px'))
                        for _ in range(9)]
        for i, button in enumerate(self.buttons):
            button.style.font_weight = 'bold'
            button.on_click(lambda b, i=i: self.click(i))
        board = ipywidgets.GridBox(self.buttons, layout=ipywidgets.Layout(
            grid_template_columns='repeat(3, 64px)', grid_gap='4px', width='204px'))
        self.message = ipywidgets.HTML()
        new_button = ipywidgets.Button(description='New puzzle', icon='random')
        new_button.on_click(lambda b: self.new())
        restart_button = ipywidgets.Button(description='Restart', icon='undo')
        restart_button.on_click(lambda b: self.start(self.initial))
        super().__init__([board, ipywidgets.HBox([new_button, restart_button]), self.message])
        self.add_class('aima-eight-puzzle-game')
        self.start(initial or scrambled_eight_puzzle(scramble, self.goal, self.rng.random()))

    def new(self):
        """Start a new random puzzle."""
        self.start(scrambled_eight_puzzle(self.scramble, self.goal, self.rng.random()))

    def start(self, state):
        """Start (or restart) the game from the given state."""
        self.initial, self.state, self.moves, self.solved = tuple(state), tuple(state), 0, False
        self.draw('Click a tile next to the blank to slide it.')

    def draw(self, text):
        """Show the current state and a message."""
        for button, tile in zip(self.buttons, self.state):
            button.description = str(tile) if tile else '\u00a0'  # a blank label may not refresh
            button.style.button_color = '#555555' if tile == 0 else COLORS['tile']
        self.message.value = f'Moves: {self.moves}. {text}'

    def click(self, i):
        """Slide tile i into the blank if they are adjacent, else refuse."""
        if self.solved:
            self.draw('Solved already: press New puzzle to play again.')
            return
        blank = self.state.index(0)
        if i == blank:
            self.draw('That is the blank: click a tile next to it.')
            return
        if abs(i // 3 - blank // 3) + abs(i % 3 - blank % 3) != 1:
            self.draw(f'Tile {self.state[i]} is not next to the blank, so it cannot move.')
            return
        state = list(self.state)
        state[blank], state[i] = state[i], 0
        self.state, self.moves = tuple(state), self.moves + 1
        if self.state == self.goal:
            self.solved = True
            optimum = len(astar_search(EightPuzzle(self.initial, self.goal),
                                       lambda n: manhattan(n, self.goal)).solution())
            verdict = 'optimal.'
            if self.moves > optimum:
                verdict = f'{moves(self.moves - optimum)} longer than the optimum.'
            self.draw(f'<b>Solved in {moves(self.moves)}.</b> '
                      f'A* finds a solution in {moves(optimum)}, so yours is {verdict}')
        else:
            self.draw('')


# ______________________________________________________________________________
# N-queens


def n_queens_svg(state, n=None, column=None):
    """Draw an n-queens board. The state is a tuple whose c-th entry is the row of
    the queen in column c, or -1 if the column is empty (as in NQueensProblem), or
    a dict {column: row} (as a CSP assignment). Queens attacked by another queen
    are drawn in red; column, if given, is highlighted."""
    if isinstance(state, dict):
        n = n or len(state)
        queens = dict(state)
    else:
        n = n or len(state)
        queens = {c: r for c, r in enumerate(state) if r != -1}
    cell = 40
    body = []
    for r in range(n):
        for c in range(n):
            square = COLORS['light'] if (r + c) % 2 == 0 else COLORS['dark']
            body.append(f'<rect x="{c * cell}" y="{r * cell}" width="{cell}" height="{cell}" '
                        f'fill="{square}"/>')
    if column is not None:
        body.append(f'<rect x="{column * cell + 1.5}" y="1.5" width="{cell - 3}" '
                    f'height="{n * cell - 3}" fill="none" stroke="{COLORS["path"]}" '
                    f'stroke-width="3"/>')
    for c, r in queens.items():
        attacked = any(r == r2 or abs(r - r2) == abs(c - c2)
                       for c2, r2 in queens.items() if c2 != c)
        x, y = c * cell + cell / 2, r * cell + cell / 2
        fill = COLORS['attacked'] if attacked else '#333333'
        body.append(f'<circle cx="{x}" cy="{y}" r="{cell * 0.36}" fill="{fill}" '
                    f'stroke="#222222" stroke-width="1.5"/>'
                    f'<text x="{x}" y="{y + 6}" font-size="18" font-weight="bold" '
                    f'text-anchor="middle" '
                    f'fill="#ffffff">Q</text>')
    return svg_page(''.join(body), n * cell, n * cell, max(n * cell, 240))


def n_queens_player(search, problem, *args, speed=1.0, **kwargs):
    """Run ``search(problem, *args, **kwargs)`` on an NQueensProblem and return a
    Player that shows the board of each expanded state, then the solution."""
    recorder = RecordingProblem(problem)
    goal = search(recorder, *args, **kwargs)
    steps = recorder.steps(goal)

    def render(player, i):
        state = steps[i]['current']
        if state is None:
            state = problem.initial
            caption = f'Step 0 of {len(steps) - 1}: the search starts from the empty board.'
        elif 'path' in steps[i]:
            caption = f'Solution found after {len(steps[i]["explored"])} expansions.'
        else:
            placed = sum(r != -1 for r in state)
            caption = f'Step {i} of {len(steps) - 1}: expand a board with {placed} queens.'
            if goal is None and i == len(steps) - 1:
                caption += ' No solution.'
        column = None if -1 not in state else state.index(-1)
        return n_queens_svg(state, problem.N, column) + f'<div>{caption}</div>'

    return Player(steps, render, speed)
