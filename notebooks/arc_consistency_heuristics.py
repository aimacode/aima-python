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

# %% [markdown] pycharm={}
# # Constraint Satisfaction Problems
# ---
# # Heuristics for Arc-Consistency Algorithms
#
# ## Introduction
# A ***Constraint Satisfaction Problem*** is a triple $(X,D,C)$ where: 
# - $X$ is a set of variables $X_1, …, X_n$;
# - $D$ is a set of domains $D_1, …, D_n$, one for each variable and each of which consists of a set of allowable values $v_1, ..., v_k$;
# - $C$ is a set of constraints that specify allowable combinations of values.
#
# A CSP is called *arc-consistent* if every value in the domain of every variable is supported by all the neighbors of the variable while, is called *inconsistent*, if it has no solutions. <br>
# ***Arc-consistency algorithms*** remove all unsupported values from the domains of variables making the CSP *arc-consistent* or decide that a CSP is *inconsistent* by finding that some variable has no supported values in its domain. <br> 
# Heuristics significantly enhance the efficiency of the *arc-consistency algorithms* improving their average performance in terms of *consistency-checks* which can be considered a standard measure of goodness for such algorithms. *Arc-heuristic* operate at arc-level and selects the constraint that will be used for the next check, while *domain-heuristics* operate at domain-level and selects which values will be used for the next support-check.

# %%
from aima.csp import *

# %% [markdown]
# ## Domain-Heuristics for Arc-Consistency Algorithms
# In <a name="ref-1"/>[[1]](#cite-van2002domain) are investigated the effects of a *domain-heuristic* based on the notion of a *double-support check* by studying its average time-complexity.
#
# The objective of *arc-consistency algorithms* is to resolve some uncertainty; it has to be know, for each $v_i \in D_i$ and for each $v_j \in D_j$, whether it is supported.
#
# A *single-support check*, $(v_i, v_j) \in C_{ij}$, is one in which, before the check is done, it is already known that either $v_i$ or $v_j$ are supported. 
#
# A *double-support check* $(v_i, v_j) \in C_{ij}$, is one in which there is still, before the check, uncertainty about the support-status of both $v_i$ and $v_j$. 
#
# If a *double-support check* is successful, two uncertainties are resolved. If a *single-support check* is successful, only one uncertainty is resolved. A good *arc-consistency algorithm*, therefore, would always choose to do a *double-support check* in preference of a *single-support check*, because the cormer offers the potential higher payback.
#
# The improvement with *double-support check* is that, where possible, *consistency-checks* are used to find supports for two values, one value in the domain of each variable, which were previously known to be unsupported. It is motivated by the insight that *in order to minimize the number of consistency-checks it is necessary to maximize the number of uncertainties which are resolved per check*.

# %% [markdown] pycharm={}
# ### AC-3b: an improved version of AC-3 with Double-Support Checks

# %% [markdown]
# As shown in <a name="ref-2"/>[[2]](#cite-van2000improving) the idea is to use *double-support checks* to improve the average performance of `AC3` which does not exploit the fact that relations are bidirectional and results in a new general purpose *arc-consistency algorithm* called `AC3b`.

# %% pycharm={}
# %psource AC3

# %% pycharm={}
# %psource revise

# %% [markdown]
# At any stage in the process of making 2-variable CSP *arc-consistent* in `AC3b`:
# - there is a set $S_i^+ \subseteq D_i$ whose values are all known to be supported by $X_j$;
# - there is a set $S_i^? = D_i \setminus S_i^+$ whose values are unknown, as yet, to be supported by $X_j$.
#
# The same holds if the roles for $X_i$ and $X_j$ are exchanged.
#
# In order to establish support for a value $v_i^? \in S_i^?$ it seems better to try to find a support among the values in $S_j^?$ first, because for each $v_j^? \in S_j^?$ the check $(v_i^?,v_j^?) \in C_{ij}$ is a *double-support check* and it is just as likely that any $v_j^? \in S_j^?$ supports $v_i^?$ than it is that any $v_j^+ \in S_j^+$ does. Only if no support can be found among the elements in $S_j^?$, should the elements $v_j^+$ in $S_j^+$ be used for *single-support checks* $(v_i^?,v_j^+) \in C_{ij}$. After it has been decided for each value in $D_i$ whether it is supported or not, either $S_x^+ = \emptyset$ and the 2-variable CSP is *inconsistent*, or $S_x^+ \neq \emptyset$ and the CSP is *satisfiable*. In the latter case, the elements from $D_i$ which are supported by $j$ are given by $S_x^+$. The elements in $D_j$ which are supported by $x$ are given by the union of $S_j^+$ with the set of those elements of $S_j^?$ which further processing will show to be supported by some $v_i^+ \in S_x^+$.

# %% pycharm={}
# %psource AC3b

# %% pycharm={}
# %psource partition

# %% [markdown] pycharm={}
# `AC3b` is a refinement of the `AC3` algorithm which consists of the fact that if, when arc $(i,j)$ is being processed and the reverse arc $(j,i)$ is also in the queue, then consistency-checks can be saved because only support for the elements in $S_j^?$ has to be found (as opposed to support for all the elements in $D_j$ in the
# `AC3` algorithm). <br>
# `AC3b` inherits all its properties like $\mathcal{O}(ed^3)$ time-complexity and $\mathcal{O}(e + nd)$ space-complexity fron `AC3` and where $n$ denotes the number of variables in the CSP, $e$ denotes the number of binary constraints and $d$ denotes the maximum domain-size of the variables.

# %% [markdown] pycharm={}
# ## Arc-Heuristics for Arc-Consistency Algorithms

# %% [markdown] pycharm={}
# Many *arc-heuristics* can be devised, based on three major features of CSPs:
# - the number of acceptable pairs in each constraint (the *constraint size* or *satisfiability*);
# - the *domain size*;
# - the number of binary constraints that each variable participates in, equal to the *degree* of the node of that variable in the constraint graph. 
#
# Simple examples of heuristics that might be expected to improve the efficiency of relaxation are:
# - ordering the list of variable pairs by *increasing* relative *satisfiability*;
# - ordering by *increasing size of the domain* of the variable $v_j$ relaxed against $v_i$;
# - ordering by *descending degree* of node of the variable relaxed.
#
# In <a name="ref-3"/>[[3]](#cite-wallace1992ordering) are investigated the effects of these *arc-heuristics* in an empirical way, experimenting the effects of them on random CSPs. Their results demonstrate that the first two, later called `sat up` and `dom j up` for n-ary and binary CSPs respectively, significantly reduce the number of *consistency-checks*.

# %% pycharm={}
# %psource dom_j_up

# %% pycharm={}
# %psource sat_up

# %% [markdown] pycharm={}
# ## Experimental Results

# %% [markdown] pycharm={}
# For the experiments below on binary CSPs, in addition to the two *arc-consistency algorithms* already cited above, `AC3` and `AC3b`, the `AC4` algorithm was used. <br>
# The `AC4` algorithm runs in $\mathcal{O}(ed^2)$ worst-case time but can be slower than `AC3` on average cases.

# %% pycharm={}
# %psource AC4

# %% [markdown]
# ### Sudoku

# %% [markdown] pycharm={}
# #### Easy Sudoku

# %% pycharm={}
sudoku = Sudoku(easy1)
sudoku.display(sudoku.infer_assignment())

# %% pycharm={}
# %time _, checks = AC3(sudoku, arc_heuristic=no_arc_heuristic)
f'AC3 needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(easy1)
# %time _, checks = AC3b(sudoku, arc_heuristic=no_arc_heuristic)
f'AC3b needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(easy1)
# %time _, checks = AC4(sudoku, arc_heuristic=no_arc_heuristic)
f'AC4 needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(easy1)
# %time _, checks = AC3(sudoku, arc_heuristic=dom_j_up)
f'AC3 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(easy1)
# %time _, checks = AC3b(sudoku, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(easy1)
# %time _, checks = AC4(sudoku, arc_heuristic=dom_j_up)
f'AC4 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %%
backtracking_search(sudoku, select_unassigned_variable=mrv, inference=forward_checking)
sudoku.display(sudoku.infer_assignment())

# %% [markdown] pycharm={}
# #### Harder Sudoku

# %% pycharm={}
sudoku = Sudoku(harder1)
sudoku.display(sudoku.infer_assignment())

# %% pycharm={}
# %time _, checks = AC3(sudoku, arc_heuristic=no_arc_heuristic)
f'AC3 needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(harder1)
# %time _, checks = AC3b(sudoku, arc_heuristic=no_arc_heuristic)
f'AC3b needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(harder1)
# %time _, checks = AC4(sudoku, arc_heuristic=no_arc_heuristic)
f'AC4 needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(harder1)
# %time _, checks = AC3(sudoku, arc_heuristic=dom_j_up)
f'AC3 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(harder1)
# %time _, checks = AC3b(sudoku, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
sudoku = Sudoku(harder1)
# %time _, checks = AC4(sudoku, arc_heuristic=dom_j_up)
f'AC4 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
backtracking_search(sudoku, select_unassigned_variable=mrv, inference=forward_checking)
sudoku.display(sudoku.infer_assignment())

# %% [markdown] pycharm={}
# ### 8 Queens

# %% pycharm={}
chess = NQueensCSP(8)
chess.display(chess.infer_assignment())

# %% pycharm={}
# %time _, checks = AC3(chess, arc_heuristic=no_arc_heuristic)
f'AC3 needs {checks} consistency-checks'

# %% pycharm={}
chess = NQueensCSP(8)
# %time _, checks = AC3b(chess, arc_heuristic=no_arc_heuristic)
f'AC3b needs {checks} consistency-checks'

# %% pycharm={}
chess = NQueensCSP(8)
# %time _, checks = AC4(chess, arc_heuristic=no_arc_heuristic)
f'AC4 needs {checks} consistency-checks'

# %% pycharm={}
chess = NQueensCSP(8)
# %time _, checks = AC3(chess, arc_heuristic=dom_j_up)
f'AC3 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
chess = NQueensCSP(8)
# %time _, checks = AC3b(chess, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
chess = NQueensCSP(8)
# %time _, checks = AC4(chess, arc_heuristic=dom_j_up)
f'AC4 with DOM J UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
backtracking_search(chess, select_unassigned_variable=mrv, inference=forward_checking)
chess.display(chess.infer_assignment())

# %% [markdown]
# For the experiments below on n-ary CSPs, due to the n-ary constraints, the `GAC` algorithm was used. <br>
# The `GAC` algorithm has $\mathcal{O}(er^2d^t)$ time-complexity and $\mathcal{O}(erd)$ space-complexity where $e$ denotes the number of n-ary constraints, $r$ denotes the constraint arity and $d$ denotes the maximum domain-size of the variables.

# %% pycharm={}
# %psource ACSolver.GAC

# %% [markdown] pycharm={}
# ### Crossword

# %% pycharm={}
crossword = Crossword(crossword1, words1)
crossword.display()
words1

# %% pycharm={}
# %time _, _, checks = ACSolver(crossword).GAC(arc_heuristic=no_heuristic)
f'GAC needs {checks} consistency-checks'

# %% pycharm={}
crossword = Crossword(crossword1, words1)
# %time _, _, checks = ACSolver(crossword).GAC(arc_heuristic=sat_up)
f'GAC with SAT UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
crossword.display(ACSolver(crossword).domain_splitting())

# %% [markdown] pycharm={}
# ### Kakuro

# %% [markdown]
# #### Easy Kakuro

# %% pycharm={}
kakuro = Kakuro(kakuro2)
kakuro.display()

# %% pycharm={}
# %time _, _, checks = ACSolver(kakuro).GAC(arc_heuristic=no_heuristic)
f'GAC needs {checks} consistency-checks'

# %% pycharm={}
kakuro = Kakuro(kakuro2)
# %time _, _, checks = ACSolver(kakuro).GAC(arc_heuristic=sat_up)
f'GAC with SAT UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
kakuro.display(ACSolver(kakuro).domain_splitting())

# %% [markdown] pycharm={}
# #### Medium Kakuro

# %% pycharm={}
kakuro = Kakuro(kakuro3)
kakuro.display()

# %% pycharm={}
# %time _, _, checks = ACSolver(kakuro).GAC(arc_heuristic=no_heuristic)
f'GAC needs {checks} consistency-checks'

# %% pycharm={}
kakuro = Kakuro(kakuro3)
# %time _, _, checks = ACSolver(kakuro).GAC(arc_heuristic=sat_up)
f'GAC with SAT UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
kakuro.display(ACSolver(kakuro).domain_splitting())

# %% [markdown] pycharm={}
# #### Harder Kakuro

# %% pycharm={}
kakuro = Kakuro(kakuro4)
kakuro.display()

# %% pycharm={}
# %time _, _, checks = ACSolver(kakuro).GAC()
f'GAC needs {checks} consistency-checks'

# %% pycharm={}
kakuro = Kakuro(kakuro4)
# %time _, _, checks = ACSolver(kakuro).GAC(arc_heuristic=sat_up)
f'GAC with SAT UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
kakuro.display(ACSolver(kakuro).domain_splitting())

# %% [markdown] pycharm={}
# ### Cryptarithmetic Puzzle

# %% [markdown]
# $$
# \begin{array}{@{}r@{}}
#      S E N D \\
# {} + M O R E \\
#    \hline
#    M O N E Y
# \end{array}
# $$

# %% pycharm={}
cryptarithmetic = NaryCSP(
    {'S': set(range(1, 10)), 'M': set(range(1, 10)),
     'E': set(range(0, 10)), 'N': set(range(0, 10)), 'D': set(range(0, 10)),
     'O': set(range(0, 10)), 'R': set(range(0, 10)), 'Y': set(range(0, 10)),
     'C1': set(range(0, 2)), 'C2': set(range(0, 2)), 'C3': set(range(0, 2)),
     'C4': set(range(0, 2))},
    [Constraint(('S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y'), all_diff_constraint),
     Constraint(('D', 'E', 'Y', 'C1'), lambda d, e, y, c1: d + e == y + 10 * c1),
     Constraint(('N', 'R', 'E', 'C1', 'C2'), lambda n, r, e, c1, c2: c1 + n + r == e + 10 * c2),
     Constraint(('E', 'O', 'N', 'C2', 'C3'), lambda e, o, n, c2, c3: c2 + e + o == n + 10 * c3),
     Constraint(('S', 'M', 'O', 'C3', 'C4'), lambda s, m, o, c3, c4: c3 + s + m == o + 10 * c4),
     Constraint(('M', 'C4'), eq)])

# %% pycharm={}
# %time _, _, checks = ACSolver(cryptarithmetic).GAC(arc_heuristic=no_heuristic)
f'GAC needs {checks} consistency-checks'

# %% pycharm={}
# %time _, _, checks = ACSolver(cryptarithmetic).GAC(arc_heuristic=sat_up)
f'GAC with SAT UP arc heuristic needs {checks} consistency-checks'

# %% pycharm={}
assignment = ACSolver(cryptarithmetic).domain_splitting()

from IPython.display import Latex
display(Latex(r'\begin{array}{@{}r@{}} ' + '{}{}{}{}'.format(assignment['S'], assignment['E'], assignment['N'], assignment['D']) + r' \\ + ' + 
              '{}{}{}{}'.format(assignment['M'], assignment['O'], assignment['R'], assignment['E']) + r' \\ \hline ' + 
              '{}{}{}{}{}'.format(assignment['M'], assignment['O'], assignment['N'], assignment['E'], assignment['Y']) + ' \end{array}'))

# %% [markdown] pycharm={}
# ## References
#
# <a name="cite-van2002domain"/><sup>[[1]](#ref-1) </sup>Van Dongen, Marc RC. 2002. _Domain-heuristics for arc-consistency algorithms_.
#
# <a name="cite-van2000improving"/><sup>[[2]](#ref-2) </sup>Van Dongen, MRC and Bowen, JA. 2000. _Improving arc-consistency algorithms with double-support checks_.
#
# <a name="cite-wallace1992ordering"/><sup>[[3]](#ref-3) </sup>Wallace, Richard J and Freuder, Eugene Charles. 1992. _Ordering heuristics for arc consistency algorithms_.
