# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run bootstrap.ipynb

# %% [markdown]
# # Game Theory and Social Choice (Chapter 18)
#
# Demonstrations of the non-cooperative game theory, cooperative game theory and social-choice algorithms in [`game_theory4e.py`](game_theory4e.py).

# %%
from aima.game_theory import *

# %% [markdown]
# ## 18.2 Non-cooperative game theory
#
# ### Dominant strategies in the prisoner's dilemma
# Payoffs are utilities (minus the years in prison); row 0 = *testify*, row 1 = *refuse*.

# %%
ali = [[-5, 0], [-10, -1]]
bo = [[-5, -10], [0, -1]]
print('Ali dominant strategy (0 = testify):', dominant_strategy(ali))
print('Iterated dominance leaves (rows, cols):', iterated_dominance(ali, bo))
print('Pure Nash equilibria:', pure_nash_equilibria(ali, bo))

# %% [markdown]
# ### Zero-sum games: two-finger Morra
# Solved by linear programming (von Neumann's minimax theorem). The value is $-1/12$ and the optimal mixed strategy is $[7/12, 5/12]$.

# %%
value, row, col = solve_zero_sum_game([[2, -3], [-3, 4]])
print('value =', round(value, 4), ' (exact -1/12 =', round(-1/12, 4), ')')
print('row player strategy =', [round(p, 4) for p in row])
print('matching pennies =', solve_zero_sum_game([[1, -1], [-1, 1]])[0])


# %% [markdown]
# ## 18.3 Cooperative game theory
#
# ### Shapley value and the core: the glove market
# Players 1 and 2 each hold a left glove, player 3 a right glove; a coalition is worth the number of complete pairs it can make. The scarce right glove captures most of the value.

# %%
def gloves(coalition):
    return min(len({1, 2} & coalition), len({3} & coalition))

phi = shapley_value([1, 2, 3], gloves)
print('Shapley value:', {k: round(v, 3) for k, v in phi.items()})
print('(0, 0, 1) in the core?', is_in_core([1, 2, 3], gloves, {1: 0, 2: 0, 3: 1}))
print('(0.5, 0, 0.5) in the core?', is_in_core([1, 2, 3], gloves, {1: 0.5, 2: 0, 3: 0.5}))

# %% [markdown]
# ## 18.4 Making collective decisions
#
# ### Voting rules can disagree
# An election with 4 voters `a>b>c`, 3 voters `b>c>a`, 2 voters `c>b>a`: plurality, Borda count and the Condorcet winner all pick different (or no) winners.

# %%
election = [['a', 'b', 'c']] * 4 + [['b', 'c', 'a']] * 3 + [['c', 'b', 'a']] * 2
print('plurality:', plurality_winner(election))
print('Borda    :', borda_winner(election))
print('Condorcet:', condorcet_winner(election))
paradox = [['a', 'b', 'c'], ['c', 'a', 'b'], ['b', 'c', 'a']]
print("Condorcet's paradox ->", condorcet_winner(paradox))

# %% [markdown]
# ### Auctions, contract net and bargaining

# %%
print('Vickrey winner, price:', vickrey_auction({'a': 10, 'b': 8, 'c': 5}))

costs = {('painter', 'paint'): 10, ('cheap_painter', 'paint'): 7, ('electrician', 'wire'): 5}
print('Contract net:', contract_net(['paint', 'wire'],
      ['painter', 'cheap_painter', 'electrician'],
      bid=lambda agent, task: costs.get((agent, task))))

print('Bargaining (equal patience 0.8):', alternating_offers_bargaining(0.8, 0.8))
print('Bargaining (A more patient)    :', alternating_offers_bargaining(0.9, 0.5))
