# OK, the following are not as widely useful utilities as some of the other
# functions here, but they do show up wherever we have 2D grids: Wumpus and
# Vacuum worlds, TicTacToe and Checkers, and Markov Decision Processes.
# __________________________________________________________________________
import math

from utils import clip

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return math.hypot((xA - xB), (yA - yB))


def distance_squared(a, b):
    """The square of the distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return (xA - xB)**2 + (yA - yB)**2


def vector_clip(vector, lowest, highest):
    """Return vector, except if any element is less than the corresponding
    value of lowest or more than the corresponding value of highest, clip to
    those values."""
    return type(vector)(map(clip, vector, lowest, highest))
