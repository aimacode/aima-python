"""Multiagent decision making: game theory and social choice (Chapter 18)."""

from collections import Counter, defaultdict
from itertools import combinations, permutations
from math import factorial

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


# ______________________________________________________________________________
# 18.3 Cooperative Game Theory


def shapley_value(players, characteristic_function):
    """
    [Section 18.3]
    The Shapley value of a cooperative game (players, v): a fair division of the
    grand coalition's value v(N) that pays each player the average, over all n!
    orderings of the players, of the marginal contribution
    mc_i(C) = v(C and {i}) - v(C) they make to the players preceding them. The
    'characteristic_function' is a callable mapping a frozenset of players to its
    value. Returns a dict mapping each player to their Shapley value.
    """
    players = list(players)
    phi = {i: 0.0 for i in players}
    for order in permutations(players):
        preceding = set()
        for i in order:
            phi[i] += (characteristic_function(frozenset(preceding | {i})) -
                       characteristic_function(frozenset(preceding)))
            preceding.add(i)
    return {i: phi[i] / factorial(len(players)) for i in players}


def is_in_core(players, characteristic_function, payoff):
    """
    [Section 18.3]
    True if 'payoff' (a dict mapping each player to their share) lies in the core
    of the cooperative game (players, v): it must distribute exactly the grand
    coalition's value (sum of shares = v(N)) and be immune to defection, i.e. for
    every coalition C the players in C receive at least v(C) (otherwise C would be
    better off on its own). An empty core means the grand coalition cannot form.
    """
    players = list(players)
    # efficiency: the grand coalition's value is fully distributed
    if not np.isclose(sum(payoff[i] for i in players), characteristic_function(frozenset(players))):
        return False
    # no coalition can object: x(C) >= v(C) for every coalition C
    return all(sum(payoff[i] for i in coalition) >= characteristic_function(frozenset(coalition)) - 1e-9
               for size in range(1, len(players)) for coalition in combinations(players, size))


# ______________________________________________________________________________
# 18.4 Making Collective Decisions


def plurality_winner(preferences):
    """
    [Section 18.4]
    Winner under plurality voting: the candidate ranked first by the most voters.
    'preferences' is a list of ballots, each a list of candidates ordered from
    most to least preferred.
    """
    first_choices = Counter(ballot[0] for ballot in preferences)
    return max(first_choices, key=first_choices.get)


def borda_winner(preferences):
    """
    [Section 18.4]
    Winner under the Borda count: with k candidates each voter awards k points to
    their top choice, k-1 to the next, down to 1 for the last; the candidate with
    the highest total score wins.
    """
    scores = defaultdict(int)
    for ballot in preferences:
        for rank, candidate in enumerate(ballot):
            scores[candidate] += len(ballot) - rank
    return max(scores, key=scores.get)


def condorcet_winner(preferences):
    """
    [Section 18.4]
    The Condorcet winner: the candidate that beats every other candidate in a
    pairwise majority comparison. Returns None when no such candidate exists
    (Condorcet's paradox), in which case majority preference is cyclic.
    """
    candidates = list(preferences[0])

    def beats(a, b):  # a majority of voters rank a above b
        return sum(ballot.index(a) < ballot.index(b) for ballot in preferences) > len(preferences) / 2

    return next((a for a in candidates if all(a == b or beats(a, b) for b in candidates)), None)


def vickrey_auction(bids):
    """
    [Section 18.4]
    Sealed-bid second-price (Vickrey) auction: the highest bidder wins but pays
    only the second-highest bid. 'bids' maps each bidder to their bid. Returns the
    (winner, price) pair. Because the winner does not pay their own bid, bidding
    one's true value is a dominant strategy (the mechanism is truth-revealing).
    """
    ranked = sorted(bids.values(), reverse=True)
    winner = max(bids, key=bids.get)
    price = ranked[1] if len(ranked) > 1 else ranked[0]
    return winner, price


def contract_net(tasks, agents, bid, select=min):
    """
    [Section 18.4.1]
    The contract net protocol for task allocation. For each task the manager
    broadcasts a task announcement; every agent submits a bid through the callable
    bid(agent, task), which returns a numeric bid or None when the agent cannot or
    will not perform the task. The manager then awards the task to the agent with
    the best bid ('select' = min for costs, max for values). Returns a dict mapping
    each task to an (agent, bid) award, or to None if nobody bid.
    """
    allocation = {}
    for task in tasks:
        bids = {agent: bid(agent, task) for agent in agents}
        bids = {agent: b for agent, b in bids.items() if b is not None}
        allocation[task] = (select(bids, key=bids.get), select(bids.values())) if bids else None
    return allocation


def alternating_offers_bargaining(discount_a, discount_b):
    """
    [Section 18.4.4]
    Rubinstein's alternating-offers bargaining over how to split a pie of size 1
    between two impatient agents with discount factors discount_a and discount_b
    in [0, 1). In the unique subgame-perfect equilibrium the agent who makes the
    first offer (A) keeps a share (1 - discount_b) / (1 - discount_a * discount_b)
    and B receives the rest; the more patient an agent is (larger discount factor)
    the larger the share it secures. Returns the (share_a, share_b) pair.
    """
    share_a = (1 - discount_b) / (1 - discount_a * discount_b)
    return share_a, 1 - share_a
