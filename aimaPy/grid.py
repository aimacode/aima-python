# OK, the following are not as widely useful utilities as some of the other
# functions here, but they do show up wherever we have 2D grids: Wumpus and
# Vacuum worlds, TicTacToe and Checkers, and markov decision Processes.
# __________________________________________________________________________
import math


orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, -1)


def turn_left(heading):
    return turn_heading(heading, +1)


def distance(a, b):
    """The distance between two (x, y) points.
        >>> distance((1,2),(5,5))
            5.0 
      """
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


def distance_squared(a, b):
    """The square of the distance between two (x, y) points.
       >>> distance_squared((1,2),(5,5))
           25.0
    """
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def distance2(a, b):
    "The square of the distance between two (x, y) points."
    return distance_squared(a, b)


def clip(x, lowest, highest):
    """Return x clipped to the range [lowest..highest].
    >>> [clip(x, 0, 1) for x in [-1, 0.5, 10]]
    [0, 0.5, 1]
    """
    return max(lowest, min(x, highest))


def vector_clip(vector, lowest, highest):
    """Return vector, except if any element is less than the corresponding
    value of lowest or more than the corresponding value of highest, clip to
    those values.
    >>> vector_clip((-1, 10), (0, 0), (9, 9))
    (0, 9)
    """
    return type(vector)(list(map(clip, vector, lowest, highest)))
