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
    csp_player            a CSP solver, one step per assignment, backtrack or pruning

The CSP solvers are recorded with record_csp (in aima.csp). EightPuzzleGame lets
the user solve an 8-puzzle by clicking the tiles, and MapColoringGame color a map.
game_tree_player replays the tree that minimax or alpha-beta explores (recorded
with RecordingGame, in aima.games), and TicTacToeGame plays against them.
plan_player executes a plan one action at a time, and graph_plan_player shows
the planning graph that GraphPlan builds, level by level. dpll_player replays
DPLL (traced with traced_dpll_satisfiable, in aima.logic), and WumpusWorldGame
lets the user explore the wumpus world while a KB says which squares are safe.
"""

import copy
import itertools
import math
import random
from html import escape

import ipywidgets

from aima.csp import NQueensCSP, record_csp
from aima.games import RecordingGame, TicTacToe, alpha_beta_search, minmax_decision
from aima.logic import (associate, breeze, conjuncts, dpll_satisfiable, find_pure_symbol,
                        find_unit_clause, pit, pl_true, prop_symbols, stench, to_cnf,
                        traced_dpll_satisfiable, wumpus)
from aima.planning import GraphPlan
from aima.utils import Expr
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


# the tiles that can move are outlined on hover; the style is scoped to the games
_GAME_STYLE = (
    '<style>'
    '.aima-game .aima-tile {font-size: 28px; font-weight: bold; border-radius: 6px;'
    ' cursor: default;}'
    '.aima-game .aima-movable {cursor: pointer;}'
    '.aima-game .aima-movable:hover {box-shadow: inset 0 0 0 3px #2a9d8f;}'
    '.aima-game .aima-blank {box-shadow: inset 0 0 0 2px #d0d0d0;}'
    '</style>')


class EightPuzzleGame(ipywidgets.VBox):
    """An 8-puzzle the user solves by clicking a tile next to the blank. Any other
    click is refused with a message; when the puzzle is solved the number of
    moves is compared with the optimum found by A* (Manhattan heuristic)."""

    def __init__(self, initial=None, scramble=16, seed=None):
        self.goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.scramble, self.rng = scramble, random.Random(seed)
        self.buttons = [ipywidgets.Button(layout=ipywidgets.Layout(width='80px', height='80px'))
                        for _ in range(9)]
        for i, button in enumerate(self.buttons):
            button.add_class('aima-tile')
            button.on_click(lambda b, i=i: self.click(i))
        board = ipywidgets.GridBox(self.buttons, layout=ipywidgets.Layout(
            grid_template_columns='repeat(3, 80px)', grid_gap='4px', width='252px',
            padding='4px', border='1px solid #d0d0d0'))
        self.message = ipywidgets.HTML()
        new_button = ipywidgets.Button(description='New puzzle', icon='random',
                                       layout=ipywidgets.Layout(width='124px'))
        new_button.on_click(lambda b: self.new())
        restart_button = ipywidgets.Button(description='Restart', icon='undo',
                                           layout=ipywidgets.Layout(width='124px'))
        restart_button.on_click(lambda b: self.start(self.initial))
        super().__init__([ipywidgets.HTML(_GAME_STYLE), board,
                          ipywidgets.HBox([new_button, restart_button]), self.message])
        self.add_class('aima-eight-puzzle-game')
        self.add_class('aima-game')
        self.start(initial or scrambled_eight_puzzle(scramble, self.goal, self.rng.random()))

    def new(self):
        """Start a new random puzzle."""
        self.start(scrambled_eight_puzzle(self.scramble, self.goal, self.rng.random()))

    def start(self, state):
        """Start (or restart) the game from the given state."""
        self.initial, self.state, self.moves, self.solved = tuple(state), tuple(state), 0, False
        self.draw('click a tile next to the blank')

    def movable(self, i):
        """Is the tile in square i next to the blank?"""
        blank = self.state.index(0)
        return abs(i // 3 - blank // 3) + abs(i % 3 - blank % 3) == 1

    def draw(self, text):
        """Show the current state and a one-line status."""
        for i, (button, tile) in enumerate(zip(self.buttons, self.state)):
            button.description = str(tile) if tile else '\u00a0'  # a blank label may not refresh
            button.style.button_color = '#fafafa' if tile == 0 else COLORS['tile']
            for name, on in (('aima-blank', tile == 0),
                             ('aima-movable', tile != 0 and not self.solved and self.movable(i))):
                if on:
                    button.add_class(name)
                else:
                    button.remove_class(name)
        self.message.value = f'Moves: {self.moves} &middot; {text}'

    def click(self, i):
        """Slide tile i into the blank if they are adjacent, else refuse."""
        if self.solved:
            self.draw('solved, press New puzzle to play again')
            return
        if self.state[i] == 0:
            self.draw('that is the blank, click a tile next to it')
            return
        if not self.movable(i):
            self.draw(f'tile {self.state[i]} is not next to the blank, so it cannot move')
            return
        blank = self.state.index(0)
        state = list(self.state)
        state[blank], state[i] = state[i], 0
        self.state, self.moves = tuple(state), self.moves + 1
        if self.state == self.goal:
            self.solved = True
            optimum = len(astar_search(EightPuzzle(self.initial, self.goal),
                                       lambda n: manhattan(n, self.goal)).solution())
            verdict = 'optimal'
            if self.moves > optimum:
                verdict = f'{moves(self.moves - optimum)} longer than the optimum'
            self.draw(f'<b>solved</b>, A* needs {moves(optimum)}, so yours is {verdict}')
        else:
            self.draw('click a tile next to the blank')


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


# ______________________________________________________________________________
# Constraint satisfaction


# where to draw the regions of australia_csp (Figure 6.1)
AUSTRALIA_LAYOUT = dict(WA=(70, 150), NT=(185, 70), SA=(200, 175), Q=(310, 90),
                        NSW=(320, 195), V=(285, 270), T=(300, 345))

# the fill of each value of a map coloring
VALUE_COLORS = dict(R='#e15759', G='#59a14f', B='#4e79a7', Y='#edc948')


def circle_layout(variables, center=(200, 190), radius=150):
    """Place the variables evenly on a circle (for CSPs with no layout of their own)."""
    n = len(variables)
    return {v: (center[0] + radius * math.cos(2 * math.pi * i / n),
                center[1] + radius * math.sin(2 * math.pi * i / n))
            for i, v in enumerate(variables)}


def csp_frames(csp, events):
    """Replay the events recorded by record_csp into frames: dicts with the
    'assignment' and the remaining 'domains' after each 'event' (None at first)."""
    assignment, domains = {}, {v: list(csp.domains[v]) for v in csp.variables}
    frames = [dict(assignment={}, domains={v: list(d) for v, d in domains.items()},
                   event=None)]
    for event in events:
        kind, var = event[0], event[1]
        if kind == 'assign':
            assignment[var] = event[2]
        elif kind == 'unassign':
            assignment.pop(var, None)
        elif kind == 'suppose':
            domains[var] = [event[2]]
        elif kind == 'prune':
            if event[2] in domains[var]:
                domains[var].remove(event[2])
        else:  # restore
            for v, value in event[1]:
                if value not in domains[v]:
                    domains[v].append(value)
            for v in domains:  # keep the order of the original domains
                domains[v] = [value for value in csp.domains[v] if value in domains[v]]
        frames.append(dict(assignment=dict(assignment), event=event,
                           domains={v: list(d) for v, d in domains.items()}))
    return frames


def describe_csp_event(event):
    """One line of text for an event of record_csp."""
    if event is None:
        return 'Start: nothing assigned, every value in every domain.'
    kind = event[0]
    if kind == 'assign':
        return f'assign <b>{event[1]} = {event[2]}</b>'
    if kind == 'unassign':
        return f'unassign <b>{event[1]}</b> (backtrack)'
    if kind == 'suppose':
        return f'suppose <b>{event[1]} = {event[2]}</b>, setting aside its other values'
    if kind == 'prune':
        return f'prune <b>{event[2]}</b> from the domain of <b>{event[1]}</b>'
    return f'restore {len(event[1])} pruned values (undo an inference)'


def map_coloring_svg(csp, frame, layout=None, highlight=None):
    """Draw the constraint graph of a map-coloring CSP: each variable is filled
    with its assigned value, and the small squares under it are the values still
    in its domain. The variable in highlight is outlined."""
    layout = layout or circle_layout(csp.variables)
    assignment, domains = frame['assignment'], frame['domains']
    body = []
    drawn = set()
    for a in csp.variables:
        for b in csp.neighbors[a]:
            if b in layout and (b, a) not in drawn:
                drawn.add((a, b))
                (x1, y1), (x2, y2) = layout[a], layout[b]
                body.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                            f'stroke="{COLORS["edge"]}" stroke-width="2"/>')
    for var in csp.variables:
        x, y = layout[var]
        fill = '#ffffff'
        if var in assignment:
            fill = VALUE_COLORS.get(assignment[var], '#bbbbbb')
        ring, ring_width = ('#222222', 4) if var == highlight else ('#555555', 1.5)
        body.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="24" fill="{fill}" stroke="{ring}" '
                    f'stroke-width="{ring_width}"/>')
        body.append(f'<text x="{x:.0f}" y="{y + 5:.0f}" font-size="14" font-weight="bold" '
                    f'text-anchor="middle" fill="{COLORS["text"]}" stroke="#ffffff" '
                    f'stroke-width="3" paint-order="stroke">{escape(str(var))}</text>')
        values = csp.domains[var]
        for k, value in enumerate(values):
            vx = x - 8 * len(values) + 16 * k + 2
            color = VALUE_COLORS.get(value, '#888888') if value in domains[var] else '#ffffff'
            body.append(f'<rect x="{vx:.0f}" y="{y + 29:.0f}" width="12" height="12" '
                        f'fill="{color}" stroke="#888888"/>')
    xs = [layout[var][0] for var in csp.variables]
    ys = [layout[var][1] for var in csp.variables]
    return svg_page(''.join(body), int(max(xs) + 60), int(max(ys) + 60), 460)


def csp_player(solver, csp, *args, layout=None, speed=1.0, **kwargs):
    """Run ``solver(csp, *args, **kwargs)`` and return a Player that replays every
    assignment, backtrack and pruning it makes. An NQueensCSP is drawn as a
    board, any other CSP as its constraint graph (use layout to place the
    variables; australia_csp-like maps get AUSTRALIA_LAYOUT)."""
    events = record_csp(csp)
    solution = solver(csp, *args, **kwargs)
    frames = csp_frames(csp, events)
    queens = isinstance(csp, NQueensCSP)
    if layout is None and set(csp.variables) <= set(AUSTRALIA_LAYOUT):
        layout = AUSTRALIA_LAYOUT

    def render(player, i):
        frame = frames[i]
        event = frame['event']
        if queens:
            svg = n_queens_svg(frame['assignment'], len(csp.variables))
        else:
            var = event[1] if event and event[0] != 'restore' else None
            svg = map_coloring_svg(csp, frame, layout, var)
        caption = f'Step {i} of {len(frames) - 1}: {describe_csp_event(event)}.'
        if i == len(frames) - 1:
            if isinstance(solution, tuple):  # what AC3 returns
                caption += (' Arc-consistent.' if solution[0] else
                            ' Inconsistent: a domain is empty.')
            elif solution:
                caption += f' Solved with {csp.nassigns} assignments.'
            else:
                caption += ' No solution.'
        return f'{svg}<div style="max-width:460px">{caption}</div>'

    return Player(frames, render, speed)


class MapColoringGame(ipywidgets.VBox):
    """Color a map-coloring CSP by hand: pick a color, then click a region. A
    color that a neighboring region already has is refused, with the reason."""

    def __init__(self, csp, layout=None):
        self.csp = csp
        if layout is None:
            australia = set(csp.variables) <= set(AUSTRALIA_LAYOUT)
            layout = AUSTRALIA_LAYOUT if australia else circle_layout(csp.variables)
        self.positions = layout
        values = list(csp.domains[csp.variables[0]])
        self.color = ipywidgets.ToggleButtons(options=values, description='Color')
        self.regions = []
        for var in csp.variables:
            button = ipywidgets.Button(description=str(var),
                                       layout=ipywidgets.Layout(width='64px'))
            button.on_click(lambda b, var=var: self.click(var))
            self.regions.append(button)
        reset_button = ipywidgets.Button(description='Reset', icon='undo')
        reset_button.on_click(lambda b: self.reset())
        self.view = ipywidgets.HTML()
        self.message = ipywidgets.HTML()
        buttons = ipywidgets.HBox(self.regions + [reset_button],
                                  layout=ipywidgets.Layout(flex_flow='row wrap'))
        super().__init__([self.color, buttons, self.view, self.message])
        self.add_class('aima-map-coloring-game')
        self.reset()

    def reset(self):
        """Clear every region."""
        self.assignment = {}
        self.draw('pick a color, then click a region')

    def draw(self, text, highlight=None):
        """Show the map and a one-line status."""
        domains = {v: list(self.csp.domains[v]) for v in self.csp.variables}
        frame = dict(assignment=self.assignment, domains=domains)
        self.view.value = map_coloring_svg(self.csp, frame, self.positions, highlight)
        for button, var in zip(self.regions, self.csp.variables):
            button.style.button_color = VALUE_COLORS.get(self.assignment.get(var), '#eeeeee')
        colored = f'{len(self.assignment)} of {len(self.csp.variables)}'
        self.message.value = f'Colored: {colored} &middot; {text}'

    def click(self, var):
        """Give var the selected color, unless a neighbor already has it."""
        value = self.color.value
        clashes = [n for n in self.csp.neighbors[var] if n in self.assignment
                   and not self.csp.constraints(var, value, n, self.assignment[n])]
        if clashes:
            self.draw(f'{var} cannot be {value}: it borders {", ".join(map(str, clashes))}, '
                      f'already {value}', highlight=var)
            return
        self.assignment[var] = value
        if len(self.assignment) == len(self.csp.variables):
            self.draw('<b>solved</b>, no two neighboring regions share a color')
        else:
            self.draw(f'{var} = {value}', highlight=var)


# ______________________________________________________________________________
# Games


def game_tree_layout(game, nodes):
    """Place the recorded nodes, and a stub for each pruned action, on a grid:
    x is the leaf order, y the depth. Return (positions, stubs), where stubs maps
    a node index to the list of (move, state) it never explored."""
    explored = {}
    for i, node in enumerate(nodes):
        if node['parent'] is not None:
            explored.setdefault(node['parent'], {})[node['move']] = i
    stubs = {i: [(a, game.game.result(node['state'], a)) for a in node['actions'] or ()
                 if a not in explored.get(i, {})] for i, node in enumerate(nodes)}
    positions, next_x = {}, [0]

    def place(i, depth):
        children = []
        for a in nodes[i]['actions'] or ():
            if a in explored.get(i, {}):
                children.append(place(explored[i][a], depth + 1))
            else:
                children.append(next_x[0])
                positions[(i, a)] = (next_x[0], depth + 1)
                next_x[0] += 1
        if not children:
            children = [next_x[0]]
            next_x[0] += 1
        positions[i] = (sum(children) / len(children), depth)
        return positions[i][0]

    place(0, 0)
    return positions, stubs


def game_tree_svg(game, nodes, values, k, maximizer, last):
    """Draw the first k recorded nodes of a game tree, with MAX nodes as up
    triangles, MIN nodes as down triangles and leaves as squares. A node shows its
    value once its subtree is done, and then its pruned actions as dashed stubs."""
    positions, stubs = game_tree_layout(game, nodes)
    width = max(x for x, y in positions.values()) + 1
    depth = max(y for x, y in positions.values()) + 1
    dx, dy, pad = 56, 90, 30

    def at(key):
        x, y = positions[key]
        return pad + x * dx + dx / 2, pad + y * dy + 20

    body = []
    done = [last[i] < k for i in range(len(nodes))]
    for i in range(1, k):
        (x1, y1), (x2, y2) = at(nodes[i]['parent']), at(i)
        body.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                    f'stroke="{COLORS["edge"]}" stroke-width="2"/>')
        body.append(f'<text x="{(x1 + x2) / 2 - 4:.0f}" y="{(y1 + y2) / 2:.0f}" font-size="11" '
                    f'fill="#777777" text-anchor="end">{escape(str(nodes[i]["move"]))}</text>')
    for i in range(k):
        if not done[i]:
            continue
        for move, state in stubs[i]:
            (x1, y1), (x2, y2) = at(i), at((i, move))
            body.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="#bbbbbb" stroke-width="2" stroke-dasharray="5,4"/>'
                        f'<text x="{x2:.0f}" y="{y2 + 4:.0f}" font-size="11" fill="#999999" '
                        f'text-anchor="middle">pruned</text>')
    for i in range(k):
        x, y = at(i)
        node = nodes[i]
        current = i == k - 1
        stroke, stroke_width = ('#222222', 3) if current else ('#555555', 1.5)
        fill = COLORS['current'] if current else (COLORS['explored'] if done[i] else '#ffffff')
        if 'utility' in node or not node['actions']:
            shape = f'<rect x="{x - 14:.0f}" y="{y - 14:.0f}" width="28" height="28" '
        elif game.to_move(node['state']) == maximizer:
            shape = (f'<polygon points="{x:.0f},{y - 18:.0f} {x - 18:.0f},{y + 13:.0f} '
                     f'{x + 18:.0f},{y + 13:.0f}" ')
        else:
            shape = (f'<polygon points="{x:.0f},{y + 18:.0f} {x - 18:.0f},{y - 13:.0f} '
                     f'{x + 18:.0f},{y - 13:.0f}" ')
        body.append(f'{shape}fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
        if done[i] and values[i] is not None:
            body.append(f'<text x="{x:.0f}" y="{y + 5:.0f}" font-size="13" font-weight="bold" '
                        f'text-anchor="middle" fill="{COLORS["text"]}">{values[i]:g}</text>')
        body.append(f'<text x="{x + 22:.0f}" y="{y + 4:.0f}" font-size="11" '
                    f'fill="#555555">{escape(str(node["state"]))}</text>')
    return svg_page(''.join(body), int(width * dx + 2 * pad), int(depth * dy + pad), 760)


def game_tree_player(search, game, state=None, speed=1.0):
    """Run search(state, game) (e.g. minmax_decision or alpha_beta_search) on a
    small game, such as Fig52Game, and return a Player that replays the tree it
    explores, one node per step: the values are backed up as the subtrees are
    done, and the actions alpha-beta never explored are shown as pruned."""
    recorder = RecordingGame(game)
    state = game.initial if state is None else state
    maximizer = game.to_move(state)
    move = search(state, recorder)
    nodes = recorder.nodes
    values = recorder.values(maximizer)
    last = list(range(len(nodes)))  # the last node visited in the subtree of each node
    for i in reversed(range(1, len(nodes))):
        parent = nodes[i]['parent']
        last[parent] = max(last[parent], last[i])
    pruned = sum(len(node['actions'] or ()) for node in nodes) - (len(nodes) - 1)

    def render(player, i):
        k = i + 1
        node = nodes[i]
        if 'utility' in node:
            text = f'leaf {escape(str(node["state"]))}: utility {values[i]:g}'
        else:
            mover = escape(str(game.to_move(node['state'])))
            text = f'expand {escape(str(node["state"]))} ({mover} to move)'
        backed = [j for j in range(len(nodes)) if last[j] == i and j != i]
        if backed:
            text += '; back up ' + ', '.join(f'{escape(str(nodes[j]["state"]))} = {values[j]:g}'
                                             for j in sorted(backed, reverse=True))
        caption = f'Step {i} of {len(nodes) - 1}: {text}.'
        if k == len(nodes):
            caption += (f' The best move is <b>{escape(str(move))}</b>, with value {values[0]:g}; '
                        f'{len(nodes)} nodes visited, {pruned} actions pruned.')
        return f'{game_tree_svg(recorder, nodes, values, k, maximizer, last)}<div>{caption}</div>'

    return Player(nodes, render, speed)


class CountingGame:
    """Delegates to a game and counts the states a search generates."""

    def __init__(self, game):
        self.game, self.count = game, 0

    def result(self, state, move):
        """Delegate to the wrapped game, counting the generated state."""
        self.count += 1
        return self.game.result(state, move)

    def __getattr__(self, attr):
        return getattr(self.game, attr)


class TicTacToeGame(ipywidgets.VBox):
    """Play tic-tac-toe as X against the computer as O, which picks its moves by
    alpha-beta search, minimax or at random; after each move the status says
    how many positions the search looked at."""

    opponents = {'alpha-beta': alpha_beta_search, 'minimax': minmax_decision,
                 'random': lambda state, game: random.choice(game.actions(state))}

    def __init__(self):
        self.game = TicTacToe()
        self.buttons = {}
        for x in range(1, 4):
            for y in range(1, 4):
                button = ipywidgets.Button(layout=ipywidgets.Layout(width='80px',
                                                                    height='80px'))
                button.add_class('aima-tile')
                button.on_click(lambda b, move=(x, y): self.click(move))
                self.buttons[(x, y)] = button
        board = ipywidgets.GridBox(list(self.buttons.values()), layout=ipywidgets.Layout(
            grid_template_columns='repeat(3, 80px)', grid_gap='4px', width='252px',
            padding='4px', border='1px solid #d0d0d0'))
        self.opponent = ipywidgets.ToggleButtons(options=list(self.opponents),
                                                 description='O plays')
        new_button = ipywidgets.Button(description='New game', icon='undo')
        new_button.on_click(lambda b: self.start())
        self.message = ipywidgets.HTML()
        super().__init__([ipywidgets.HTML(_GAME_STYLE), self.opponent, board, new_button,
                          self.message])
        self.add_class('aima-tic-tac-toe-game')
        self.add_class('aima-game')
        self.start()

    def start(self):
        """Start a new game, with X (the user) to move."""
        self.state = self.game.initial
        self.draw('you are X, click a square')

    def draw(self, text):
        """Show the board and a one-line status."""
        for move, button in self.buttons.items():
            mark = self.state.board.get(move)
            button.description = mark or ' '
            button.style.button_color = COLORS['tile'] if mark else '#fafafa'
            if mark is None and not self.game.terminal_test(self.state):
                button.add_class('aima-movable')
            else:
                button.remove_class('aima-movable')
        self.message.value = text

    def outcome(self):
        """The end of the game, in words."""
        utility = self.game.utility(self.state, 'X')
        if utility == 0:
            return '<b>draw</b>'
        return '<b>you win</b>' if utility > 0 else '<b>O wins</b>'

    def click(self, move):
        """Play X at move, then let the computer answer."""
        if self.game.terminal_test(self.state):
            self.draw(f'{self.outcome()}, press New game to play again')
            return
        if move not in self.state.moves:
            self.draw('that square is taken, pick an empty one')
            return
        self.state = self.game.result(self.state, move)
        if self.game.terminal_test(self.state):
            self.draw(self.outcome())
            return
        self.draw(f'O is thinking ({self.opponent.value})...')
        counter = CountingGame(self.game)
        reply = self.opponents[self.opponent.value](self.state, counter)
        self.state = self.game.result(self.state, reply)
        looked = f'{self.opponent.value} looked at {counter.count:,} positions'
        if self.game.terminal_test(self.state):
            self.draw(f'O plays row {reply[0]}, column {reply[1]} ({looked}); {self.outcome()}')
        else:
            self.draw(f'O plays row {reply[0]}, column {reply[1]} ({looked}); your move')


# ______________________________________________________________________________
# Planning


def is_persistence(action):
    """Is action one of the no-op (persistence) actions GraphPlan adds, e.g. PAt(...)?"""
    return action.op[0] == 'P' and action.op[1:2].isupper()


def blocks_svg(state, moved=None):
    """Draw the On(block, place) facts of a blocks-world state as stacks of blocks
    on a table; the block in moved is highlighted."""
    on = {str(fact.args[0]): str(fact.args[1]) for fact in state if fact.op == 'On'}
    stacks = []
    for block in sorted(b for b, place in on.items() if place == 'Table'):
        stack = [block]
        while True:
            above = [b for b, place in on.items() if place == stack[-1]]
            if not above:
                break
            stack.append(above[0])
        stacks.append(stack)
    size, gap = 50, 30
    height = size * max(len(on), 1) + 40
    width = max(len(stacks), 1) * (size + gap) + gap
    body = [f'<rect x="0" y="{height - 16}" width="{width}" height="16" fill="{COLORS["dark"]}"/>']
    for s, stack in enumerate(stacks):
        x = gap + s * (size + gap)
        for level, block in enumerate(stack):
            y = height - 16 - (level + 1) * size
            fill = COLORS['frontier'] if block == moved else COLORS['tile']
            body.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="4" '
                        f'fill="{fill}" stroke="#555555"/>'
                        f'<text x="{x + size / 2}" y="{y + size / 2 + 7}" font-size="20" '
                        f'font-weight="bold" text-anchor="middle" '
                        f'fill="{COLORS["text"]}">{escape(block)}</text>')
    return svg_page(''.join(body), width, height, max(width, 240))


def fluents_html(state, previous, goals):
    """List the facts of a state, the ones the last action added in green and
    the ones it deleted struck through; the goals that hold are checked."""
    facts = [str(fact) for fact in state]
    gone = [str(fact) for fact in previous or () if fact not in state]
    rows = []
    for fact in sorted(facts):
        new = previous is not None and fact not in map(str, previous)
        style = f'color:{COLORS["path"]};font-weight:bold' if new else ''
        mark = ' &#10003;' if fact in map(str, goals) else ''
        rows.append(f'<span style="{style}">{escape(fact)}{mark}</span>')
    rows += [f'<span style="text-decoration:line-through;color:#999999">{escape(fact)}</span>'
             for fact in sorted(gone)]
    return ('<div style="columns:2;max-width:560px;font-family:monospace">'
            + '<br>'.join(rows) + '</div>')


def plan_player(planning_problem, plan, speed=1.0):
    """Return a Player that executes plan (a list of actions, e.g. the
    linearize'd output of GraphPlan) on a copy of planning_problem, one action
    per step. A blocks world is drawn as stacks of blocks, any other problem as
    its list of facts."""
    problem = copy.deepcopy(planning_problem)
    states = [list(problem.initial)]
    for action in plan:
        problem.act(action)
        states.append(list(problem.initial))
    blocks = any(fact.op == 'On' for fact in states[0])
    goals = planning_problem.goals

    def render(player, i):
        state = states[i]
        done = sum(goal in state for goal in goals)
        if i == 0:
            caption = 'Step 0: the initial state.'
        else:
            caption = f'Step {i} of {len(plan)}: <b>{escape(str(plan[i - 1]))}</b>.'
        caption += f' Goals holding: {done} of {len(goals)}.'
        if blocks:
            moved = str(plan[i - 1].args[0]) if i > 0 else None
            view = blocks_svg(state, moved)
        else:
            view = fluents_html(state, states[i - 1] if i > 0 else None, goals)
        return f'{view}<div>{caption}</div>'

    return Player(states, render, speed)


def graph_plan_player(planning_problem, speed=1.0):
    """Run GraphPlan on planning_problem and return a Player that shows the
    planning graph one level at a time: the literals of each state level, the
    actions of each action level (the persistence actions are only counted) and
    the number of mutex pairs; the last step marks the actions of the plan."""
    graph_plan = GraphPlan(planning_problem)
    solution = graph_plan.execute()
    levels = graph_plan.graph.levels
    chosen = {}  # level -> the actions of the plan at that level
    if solution:
        chosen = {k: {str(action) for action in step} for k, step in enumerate(solution[0])}
    goals = {str(goal) for goal in planning_problem.goals}
    frames = list(range(len(levels))) + ([len(levels)] if solution else [])

    def column(title, items, note):
        return (f'<div style="flex:0 0 auto;padding:0 8px;border-left:1px solid #dddddd">'
                f'<div style="font-weight:bold">{title}</div>{"<br>".join(items)}'
                f'<div style="color:#777777">{note}</div></div>')

    def render(player, i):
        last = min(i, len(levels) - 1)
        mark = i == len(levels)
        columns = []
        for k in range(last + 1):
            level = levels[k]
            literals = []
            for fact in sorted(map(str, level.current_state)):
                style = 'font-weight:bold' if fact in goals else ''
                literals.append(f'<span style="{style}">{escape(fact)}</span>')
            columns.append(column(f'S{k}', literals, f'{len(level.state_mutexes)} mutex pairs'))
            if k < last:
                actions = [a for a in level.current_action_links if not is_persistence(a)]
                names = []
                for action in sorted(map(str, actions)):
                    style = ''
                    if mark and action in chosen.get(k, ()):
                        style = f'color:{COLORS["path"]};font-weight:bold'
                    names.append(f'<span style="{style}">{escape(action)}</span>')
                persistence = len(level.current_action_links) - len(actions)
                columns.append(column(f'A{k}', names, f'+ {persistence} persistence actions, '
                                                      f'{len(level.action_mutexes)} mutex pairs'))
        if mark:
            caption = ('The goals (in bold) hold, with no two of them mutex, at the last level; '
                       'the actions of the plan extracted backwards are in green.')
        elif i == 0:
            caption = 'Level 0: the initial state.'
        else:
            caption = (f'Level {i}: the actions applicable at level {i - 1} and the literals '
                       f'they make true.')
        if not solution and i == frames[-1]:
            caption += ' The graph levels off without a plan.'
        return ('<div style="display:flex;flex-wrap:wrap;row-gap:12px;font-family:monospace;'
                'font-size:12px;max-width:100%">'
                f'{"".join(columns)}</div><div>{caption}</div>')

    return Player(frames, render, speed)


# ______________________________________________________________________________
# Logic


def dpll_steps(sentence, events):
    """Turn the events of traced_dpll_satisfiable into steps: dicts with the
    'model' of each call (or of each failed call, on a backtrack), the 'depth',
    and what happened, as text ('what')."""
    clauses = conjuncts(to_cnf(sentence))
    symbols = prop_symbols(sentence)
    steps, path = [], []
    for kind, value, depth in events:
        if kind == 'call':
            path[depth:] = [value]
            if depth == 0:
                what = 'start with an empty model'
            else:
                parent = path[depth - 1]
                (symbol, truth), = [(s, v) for s, v in value.items() if s not in parent]
                remaining = [s for s in symbols if s not in parent]
                unknown = [c for c in clauses if pl_true(c, parent) is None]
                if find_pure_symbol(remaining, unknown) == (symbol, truth):
                    rule = 'pure symbol'
                elif find_unit_clause(clauses, parent) == (symbol, truth):
                    rule = 'unit clause'
                else:
                    rule = 'branch'
                what = f'{rule}: {symbol} = {truth}'
            steps.append(dict(model=value, depth=depth, what=what))
        elif value is False and depth < len(path):
            false = [c for c in clauses if pl_true(c, path[depth]) is False]
            what = (f'clause {false[0]} is false: fail' if false else
                    'every way to go on failed: fail')
            steps.append(dict(model=path[depth], depth=depth, what=what))
    return steps


def dpll_player(sentence, speed=1.0):
    """Run DPLL on a propositional sentence (traced with traced_dpll_satisfiable,
    so dpll itself is unchanged) and return a Player that shows, one call per
    step, the model being built, why the last symbol got its value (pure symbol,
    unit clause or branch), and each clause of the CNF as true, false or unknown."""
    result, events = traced_dpll_satisfiable(sentence)
    steps = dpll_steps(sentence, events)
    clauses = conjuncts(to_cnf(sentence))
    symbols = sorted(prop_symbols(sentence), key=str)

    def render(player, i):
        step = steps[i]
        model = step['model']
        chips = []
        for symbol in symbols:
            if symbol in model:
                color = COLORS['path'] if model[symbol] else COLORS['current']
                chips.append(f'<span style="background:{color};color:#ffffff;padding:2px 6px;'
                             f'border-radius:4px">{escape(str(symbol))} = {model[symbol]}</span>')
            else:
                chips.append(f'<span style="border:1px solid #bbbbbb;padding:1px 6px;'
                             f'border-radius:4px">{escape(str(symbol))}</span>')
        rows = []
        for clause in clauses:
            value = pl_true(clause, model)
            style = {True: f'color:{COLORS["path"]}', False: f'color:{COLORS["current"]};'
                     f'font-weight:bold', None: ''}[value]
            rows.append(f'<span style="{style}">{escape(str(clause))}</span>')
        caption = f'Step {i} of {len(steps) - 1} (depth {step["depth"]}): {escape(step["what"])}.'
        if i == len(steps) - 1:
            caption += ' Satisfiable.' if result else ' Unsatisfiable: no model exists.'
        return ('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0">'
                f'{"".join(chips)}</div>'
                '<div style="font-family:monospace;columns:2;max-width:640px">'
                f'{"<br>".join(rows)}</div>'
                f'<div>{caption}</div>')

    return Player(steps, render, speed)


class WumpusWorldGame(ipywidgets.VBox):
    """Explore the wumpus world of Figure 7.2 by clicking a square next to the
    agent. The agent perceives a stench next to the wumpus, a breeze next to a
    pit and a glitter on the gold; after each move a propositional KB made of the
    rules and percepts of the visited squares is asked, with DPLL, which squares
    next to them are provably safe, provably a pit or the wumpus, or unknown."""

    def __init__(self, wumpus_at=(1, 3), pits=((3, 1), (3, 3), (4, 4)), gold=(2, 3), size=4):
        self.wumpus_at, self.pits, self.gold, self.size = wumpus_at, set(pits), gold, size
        self.buttons = {}
        for y in range(size, 0, -1):
            for x in range(1, size + 1):
                button = ipywidgets.Button(layout=ipywidgets.Layout(width='80px', height='64px'))
                button.add_class('aima-tile')
                button.style.font_size = '13px'
                button.on_click(lambda b, square=(x, y): self.click(square))
                self.buttons[(x, y)] = button
        layout = ipywidgets.Layout(grid_template_columns=f'repeat({size}, 80px)', grid_gap='4px',
                                   padding='4px', border='1px solid #d0d0d0',
                                   width=f'{84 * size + 10}px')
        board = ipywidgets.GridBox([self.buttons[(x, y)] for y in range(size, 0, -1)
                                    for x in range(1, size + 1)], layout=layout)
        new_button = ipywidgets.Button(description='Start again', icon='undo')
        new_button.on_click(lambda b: self.start())
        self.message = ipywidgets.HTML()
        super().__init__([ipywidgets.HTML(_GAME_STYLE), board, new_button, self.message])
        self.add_class('aima-wumpus-world-game')
        self.start()

    def neighbors(self, square):
        """The squares next to square, inside the grid."""
        x, y = square
        return [(x + dx, y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 1 <= x + dx <= self.size and 1 <= y + dy <= self.size]

    def percepts(self, square):
        """The percepts in square."""
        found = []
        if any(n == self.wumpus_at for n in self.neighbors(square)):
            found.append('Stench')
        if any(n in self.pits for n in self.neighbors(square)):
            found.append('Breeze')
        if square == self.gold:
            found.append('Glitter')
        return found

    def kb(self):
        """The sentence the agent knows: no pit or wumpus where it has been, and the
        breeze and stench rules of the visited squares with what it perceived."""
        facts = []
        for square in self.visited:
            percepts = self.percepts(square)
            facts += [~pit(*square), ~wumpus(*square)]
            for symbol, cause, seen in ((breeze, pit, 'Breeze'), (stench, wumpus, 'Stench')):
                around = associate('|', [cause(*n) for n in self.neighbors(square)])
                facts.append(Expr('<=>', symbol(*square), around))
                facts.append(symbol(*square) if seen in percepts else ~symbol(*square))
        return associate('&', facts)

    def entails(self, kb, alpha):
        """Does kb entail alpha? It does when ``kb & ~alpha`` has no model (checked with
        DPLL)."""
        return not dpll_satisfiable(kb & ~alpha)

    def start(self):
        """Put the agent back in [1,1]."""
        self.agent, self.visited, self.over = (1, 1), {(1, 1)}, False
        self.draw('click a square next to the agent to move there')

    def draw(self, text):
        """Show the grid, what the KB proves about the squares next to the visited
        ones, and a one-line status."""
        kb = self.kb()
        frontier = {n for square in self.visited for n in self.neighbors(square)} - self.visited
        for square, button in self.buttons.items():
            label, color = ' ', '#fafafa'
            if square in self.visited:
                label = ' '.join(p[0] for p in self.percepts(square)) or 'OK'
                color = COLORS['explored']
            elif square in frontier and not self.over:
                if self.entails(kb, ~pit(*square) & ~wumpus(*square)):
                    label, color = 'safe', '#cfe8cf'
                elif self.entails(kb, pit(*square)):
                    label, color = 'pit!', '#f4b6b6'
                elif self.entails(kb, wumpus(*square)):
                    label, color = 'wumpus!', '#f4b6b6'
                else:
                    label, color = '?', COLORS['frontier']
            if square == self.agent:
                label = 'A ' + (label if label != ' ' else '')
            button.description = label
            button.style.button_color = color
        percepts = ', '.join(self.percepts(self.agent)) or 'nothing'
        here = f'[{self.agent[0]},{self.agent[1]}]'
        self.message.value = f'In {here} you perceive {percepts} &middot; {text}'

    def click(self, square):
        """Move the agent to square if it is next to it."""
        if self.over:
            self.draw('press Start again to play again')
            return
        if square not in self.neighbors(self.agent):
            self.draw('you can only move to a square next to the agent')
            return
        self.agent = square
        self.visited.add(square)
        if square in self.pits or square == self.wumpus_at:
            self.over = True
            what = 'a pit' if square in self.pits else 'the wumpus'
            self.draw(f'<b>you fell into {what}</b>; the safe squares were the green ones')
        elif square == self.gold:
            self.over = True
            self.draw('<b>you found the gold</b>')
        else:
            self.draw('S = stench, B = breeze, G = glitter; safe, pit! and wumpus! are proved '
                      'by the KB, ? is unknown')
