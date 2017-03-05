<div align="center">
  <a href="http://aima.cs.berkeley.edu/"><img src="https://raw.githubusercontent.com/aimacode/aima-python/master/images/aima_logo.png"></a><br><br>
</div>
-----------------

# `aima-python` [![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python) [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/aimacode/aima-python)


Python code for the book *Artificial Intelligence: A Modern Approach.* You can use this in conjunction with a course on AI, or for study on your own. We're looking for [solid contributors](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md) to help.

## Python 3.4

This code is in Python 3.4 (Python 3.5, also works, but Python 2.x does not). You can [install the latest Python version](https://www.python.org/downloads) or use a browser-based Python interpreter such as [repl.it](https://repl.it/languages/python3).

## Structure of the Project

When complete, this project will have Python code for all the pseudocode algorithms in the book. For each major topic, such as `logic`, we will have the following three files in the main branch:

- `logic.py`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data.
- `logic.ipynb`: A Jupyter (IPython) notebook that explains and gives examples of how to use the code.
- `tests/logic_test.py`: A lightweight test suite, using `assert` statements, designed for use with [`py.test`](http://pytest.org/latest/), but also usable on their own.

# Index of Algorithms

Here is a table of algorithms, the figure, name of the code in the book and in the repository, and the file where they are implemented in the code. This chart was made for the third edition of the book and needs to be updated for the upcoming fourth edition. Empty implementations are a good place for contributors to look for an issue. The [aima-pseudocode](https://github.com/aimacode/aima-pseudocode) project describes all the algorithms from the book.

| **Figure** | **Name (in 3<sup>rd</sup> edition)** | **Name (in repository)** | **File**
|:--------|:-------------------|:---------|:-----------|
| 2.1     | Environment        | `Environment` | [`agents.py`][agents] |
| 2.1     | Agent              | `Agent` | [`agents.py`][agents] |
| 2.3     | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`agents.py`][agents] |
| 2.7     | Table-Driven-Agent | `TableDrivenAgent` | [`agents.py`][agents] |
| 2.8     | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`agents.py`][agents] |
| 2.10    | Simple-Reflex-Agent | `SimpleReflexAgent` | [`agents.py`][agents] |
| 2.12    | Model-Based-Reflex-Agent | `ReflexAgentWithState` | [`agents.py`][agents] |
| 3       | Problem            | `Problem` | [`search.py`][search] |
| 3       | Node               | `Node` | [`search.py`][search] |
| 3       | Queue              | `Queue` | [`utils.py`][utils] |
| 3.1     | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`search.py`][search] |
| 3.2     | Romania            | `romania` | [`search.py`][search] |
| 3.7     | Tree-Search        | `tree_search` | [`search.py`][search] |
| 3.7     | Graph-Search        | `graph_search` | [`search.py`][search] |
| 3.11    | Breadth-First-Search        | `breadth_first_search` | [`search.py`][search] |
| 3.14    | Uniform-Cost-Search        | `uniform_cost_search` | [`search.py`][search] |
| 3.17    | Depth-Limited-Search | `depth_limited_search` | [`search.py`][search] |
| 3.18    | Iterative-Deepening-Search | `iterative_deepening_search` | [`search.py`][search] |
| 3.22    | Best-First-Search  | `best_first_graph_search` | [`search.py`][search] |
| 3.24    | A\*-Search        | `astar_search` | [`search.py`][search] |
| 3.26    | Recursive-Best-First-Search | `recursive_best_first_search` | [`search.py`][search] |
| 4.2     | Hill-Climbing      | `hill_climbing` | [`search.py`][search] |
| 4.5     | Simulated-Annealing | `simulated_annealing` | [`search.py`][search] |
| 4.8     | Genetic-Algorithm  | `genetic_algorithm` | [`search.py`][search] |
| 4.11    | And-Or-Graph-Search | `and_or_graph_search` | [`search.py`][search] |
| 4.21    | Online-DFS-Agent   | `online_dfs_agent` | [`search.py`][search] |
| 4.24    | LRTA\*-Agent       | `LRTAStarAgent`    | [`search.py`][search] |
| 5.3     | Minimax-Decision   | `minimax_decision` | [`games.py`][games] |
| 5.7     | Alpha-Beta-Search  | `alphabeta_search` | [`games.py`][games] |
| 6       | CSP                | `CSP` | [`csp.py`][csp] |
| 6.3     | AC-3               | `AC3` | [`csp.py`][csp] |
| 6.5     | Backtracking-Search | `backtracking_search` | [`csp.py`][csp] |
| 6.8     | Min-Conflicts      | `min_conflicts` | [`csp.py`][csp] |
| 6.11    | Tree-CSP-Solver    | `tree_csp_solver` | [`csp.py`][csp] |
| 7       | KB                 | `KB` | [`logic.py`][logic] |
| 7.1     | KB-Agent           | `KB_Agent` | [`logic.py`][logic] |
| 7.7     | Propositional Logic Sentence | `Expr` | [`logic.py`][logic] |
| 7.10    | TT-Entails         | `tt_entials` | [`logic.py`][logic] |
| 7.12    | PL-Resolution      | `pl_resolution` | [`logic.py`][logic] |
| 7.14    | Convert to CNF     | `to_cnf` | [`logic.py`][logic] |
| 7.15    | PL-FC-Entails?     | `pl_fc_resolution` | [`logic.py`][logic] |
| 7.17    | DPLL-Satisfiable?  | `dpll_satisfiable` | [`logic.py`][logic] |
| 7.18    | WalkSAT            | `WalkSAT` | [`logic.py`][logic] |
| 7.20    | Hybrid-Wumpus-Agent    |         |           |
| 7.22    | SATPlan            | `SAT_plan`  | [`logic.py`][logic] |
| 9       | Subst              | `subst` | [`logic.py`][logic] |
| 9.1     | Unify              | `unify` | [`logic.py`][logic] |
| 9.3     | FOL-FC-Ask         | `fol_fc_ask` | [`logic.py`][logic] |
| 9.6     | FOL-BC-Ask         | `fol_bc_ask` | [`logic.py`][logic] |
| 9.8     | Append             |            |              |
| 10.1    | Air-Cargo-problem    |          |
| 10.2    | Spare-Tire-Problem |          |
| 10.3    | Three-Block-Tower  |          |
| 10.7    | Cake-Problem       |          |
| 10.9    | Graphplan          |          |
| 10.13   | Partial-Order-Planner |          |
| 11.1    | Job-Shop-Problem-With-Resources |          |
| 11.5    | Hierarchical-Search |          |
| 11.8    | Angelic-Search   |          |
| 11.10   | Doubles-tennis     |          |
| 13      | Discrete Probability Distribution | `ProbDist` | [`probability.py`][probability] |
| 13.1    | DT-Agent           | `DTAgent` | [`probability.py`][probability] |
| 14.9    | Enumeration-Ask    | `enumeration_ask` | [`probability.py`][probability] |
| 14.11   | Elimination-Ask    | `elimination_ask` | [`probability.py`][probability] |
| 14.13   | Prior-Sample       | `prior_sample` | [`probability.py`][probability] |
| 14.14   | Rejection-Sampling | `rejection_sampling` | [`probability.py`][probability] |
| 14.15   | Likelihood-Weighting | `likelihood_weighting` | [`probability.py`][probability] |
| 14.16   | Gibbs-Ask           | `gibbs_ask`  | [`probability.py`][probability] |
| 15.4    | Forward-Backward   | `forward_backward` | [`probability.py`][probability] |
| 15.6    | Fixed-Lag-Smoothing | `fixed_lag_smoothing` | [`probability.py`][probability] |
| 15.17   | Particle-Filtering | `particle_filtering` | [`probability.py`][probability] |
| 16.9    | Information-Gathering-Agent |          |
| 17.4    | Value-Iteration    | `value_iteration` | [`mdp.py`][mdp] |
| 17.7    | Policy-Iteration   | `policy_iteration` | [`mdp.py`][mdp] |
| 17.7    | POMDP-Value-Iteration  |           |        |
| 18.5    | Decision-Tree-Learning | `DecisionTreeLearner` | [`learning.py`][learning] |
| 18.8    | Cross-Validation   | `cross_validation` | [`learning.py`][learning] |
| 18.11   | Decision-List-Learning | `DecisionListLearner` | [`learning.py`][learning] |
| 18.24   | Back-Prop-Learning | `BackPropagationLearner` | [`learning.py`][learning] |
| 18.34   | AdaBoost           | `AdaBoost` | [`learning.py`][learning] |
| 19.2    | Current-Best-Learning |          |
| 19.3    | Version-Space-Learning |          |
| 19.8    | Minimal-Consistent-Det |          |
| 19.12   | FOIL               |          |
| 21.2    | Passive-ADP-Agent  | `PassiveADPAgent` | [`rl.py`][rl] |
| 21.4    | Passive-TD-Agent   | `PassiveTDAgent` | [`rl.py`][rl] |
| 21.8    | Q-Learning-Agent   | `QLearningAgent` | [`rl.py`][rl] |
| 22.1    | HITS               | `HITS`  | [`nlp.py`][nlp] |
| 23      | Chart-Parse        | `Chart` | [`nlp.py`][nlp] |
| 23.5    | CYK-Parse          | `CYK_parse` | [`nlp.py`][nlp] |
| 25.9    | Monte-Carlo-Localization|       |


# Index of data structures

Here is a table of the implemented data structures, the figure, name of the implementation in the repository, and the file where they are implemented.

| **Figure** | **Name (in repository)** | **File** |
|:-----------|:-------------------------|:---------|
| 3.2    | romania_map              | [`search.py`][search] |
| 4.9    | vacumm_world             | [`search.py`][search] |
| 4.23   | one_dim_state_space      | [`search.py`][search] |
| 6.1    | australia_map            | [`search.py`][search] |
| 7.13   | wumpus_world_inference   | [`logic.py`][logic] |
| 7.16   | horn_clauses_KB          | [`logic.py`][logic] |
| 17.1   | sequential_decision_environment | [`mdp.py`][mdp] |
| 18.2   | waiting_decision_tree    | [`learning.py`][learning] |


# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is on GitHub, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Many thanks to all contributors, especially @darius, @SnShine, and @reachtarunhere.

<!---Reference Links-->
[agents]:../master/agents.py
[csp]:../master/csp.py
[games]:../master/games.py
[grid]:../master/grid.py
[learning]:../master/learning.py
[logic]:../master/logic.py
[mdp]:../master/mdp.py
[nlp]:../master/nlp.py
[planning]:../master/planning.py
[probability]:../master/probability.py
[rl]:../master/rl.py
[search]:../master/search.py
[utils]:../master/utils.py
[text]:../master/text.py
