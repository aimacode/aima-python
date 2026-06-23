"""Multiagent decision making: non-cooperative game theory (Chapter 18)."""

import numpy as np
from scipy.optimize import linprog


def dominates(payoff, s, t, strongly=True):
    """
    [Section 18.2]
    True if, for the player whose payoff matrix is 'payoff' (rows = that player's
    own strategies, columns = the opponents' joint strategies), strategy s
    dominates strategy t. Strong domination requires s to be strictly better than
    t against every opponent strategy; weak domination requires s to be no worse
    everywhere and strictly better in at least one column.
    """
    payoff = np.asarray(payoff, dtype=float)
    if strongly:
        return bool(np.all(payoff[s] > payoff[t]))
    return bool(np.all(payoff[s] >= payoff[t]) and np.any(payoff[s] > payoff[t]))


def dominant_strategy(payoff, strongly=True):
    """
    [Section 18.2]
    Return the strategy (row index) that dominates all of the player's other
    strategies, or None if the player has no such dominant strategy. To analyze
    the column player of a game, pass the transpose of their payoff matrix so
    that their strategies become the rows.
    """
    payoff = np.asarray(payoff, dtype=float)
    for s in range(payoff.shape[0]):
        if all(dominates(payoff, s, t, strongly) for t in range(payoff.shape[0]) if t != s):
            return s
    return None


def pure_nash_equilibria(payoff1, payoff2):
    """
    [Section 18.2]
    All pure-strategy Nash equilibria of a two-player game, given the payoff
    matrices payoff1[i][j] and payoff2[i][j] for the row and column player when
    the row player plays i and the column player plays j. A profile (i, j) is a
    Nash equilibrium when each player is playing a best response to the other, so
    neither can gain by deviating unilaterally. Returns a list of (i, j) profiles
    (possibly empty, e.g. for matching pennies).
    """
    payoff1, payoff2 = np.asarray(payoff1, dtype=float), np.asarray(payoff2, dtype=float)
    m, n = payoff1.shape
    return [(i, j) for i in range(m) for j in range(n)
            # row player best-responds within column j and column player within row i
            if payoff1[i, j] >= payoff1[:, j].max() and payoff2[i, j] >= payoff2[i, :].max()]


def solve_zero_sum_game(payoff):
    """
    [Section 18.2]
    Solve a two-player zero-sum game by linear programming. 'payoff' is the payoff
    to the row (maximizing) player; the column (minimizing) player receives its
    negation. By von Neumann's minimax theorem the game has a value v and optimal
    mixed strategies x, y such that x maximizes the row player's guaranteed payoff
    (for every column j, sum_i x_i payoff[i][j] >= v) and y minimizes the column
    player's guaranteed loss. Returns (value, row_strategy, col_strategy).
    """
    payoff = np.asarray(payoff, dtype=float)
    m, n = payoff.shape

    # row player: maximize v s.t. for every column j, sum_i x_i payoff[i][j] >= v;
    # variables are [x_1, ..., x_m, v], and linprog minimizes, so we minimize -v
    res = linprog(c=np.append(np.zeros(m), -1),
                  A_ub=np.column_stack([-payoff.T, np.ones(n)]), b_ub=np.zeros(n),
                  A_eq=np.append(np.ones(m), 0).reshape(1, -1), b_eq=[1],
                  bounds=[(0, None)] * m + [(None, None)])
    value, row_strategy = res.x[m], res.x[:m]

    # column player: minimize w s.t. for every row i, sum_j payoff[i][j] y_j <= w
    res = linprog(c=np.append(np.zeros(n), 1),
                  A_ub=np.column_stack([payoff, -np.ones(m)]), b_ub=np.zeros(m),
                  A_eq=np.append(np.ones(n), 0).reshape(1, -1), b_eq=[1],
                  bounds=[(0, None)] * n + [(None, None)])
    col_strategy = res.x[:n]

    return value, row_strategy, col_strategy
