# ![](https://github.com/aimacode/aima-java/blob/gh-pages/aima3e/images/aima3e.jpg)`aima-python`[![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python)


Python code for the book *Artificial Intelligence: A Modern Approach.* We're loooking for one student sponsored by Google Summer of Code ([GSoC](https://summerofcode.withgoogle.com/)) to work on this project; if you want to be that student, make some good [contributions](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md) here by looking through the [Issues](https://github.com/aimacode/aima-python/issues) and resolving some), and submit an [application](https://summerofcode.withgoogle.com/terms/student). (However, be warned that we've had over 150 students express interest, so competition will be tough.) And we're always [looking for solid contributors](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md) who are not affiliated with GSoC. A big thank you to everyone who has contributed!

## Python 3.4

This code is in Python 3.4. (Of course, the current version, Python 3.5, also works.) You can [install the latest Python version](https://www.python.org/downloads), and if that doesn't work, use a browser-based Python interpreter such as [repl.it](https://repl.it/languages/python3).

## Structure of the Project

When complete, this project will have Python code for all the pseudocode algorithms in the book. For each major topic, such as `logic`, we will have the following three files in the main branch:

- `logic.py`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data.
- `logic.ipynb`: A Jupyter notebook that explains and gives examples of how to use the code.
- `tests/logic_test.py`: A lightweight test suite, using `assert` statements, designed for use with [`py.test`](http://pytest.org/latest/).



# Index of Code #

Here is a table of algorithms, the figure and page where they appear in the book, and the file where they appear in the code. Unfortuately, this chart was made for the old second edition; and has only been partially upfdated to third edition, and not at all to fourth edition. We could use help fixing up the table, based on the figures in [algorithms.pdf](https://github.com/aimacode/aima-pseudocode/blob/master/algorithms.pdf). Empty implementations are a good place for contributors to look for an issue.


| **Fig** | **Name (in 3<sup>rd</sup> edition)** | **Name (in code)** | **File**
|:--------|:-------------------|:---------|:-----------|
| 2.1     | Environment        | `Environment` | [`agents.py`](../master/agents.py) |
| 2.1     | Agent              | `Agent` | [`agents.py`](../master/agents.py) |
| 2.3     | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.7     | Table-Driven-Agent | `TableDrivenAgent` | [`agents.py`](../master/agents.py) |
| 2.8     | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.10    | Simple-Reflex-Agent | `SimpleReflexAgent` | [`agents.py`](../master/agents.py) |
| 2.12    | Model-Based-Reflex-Agent | `ReflexAgentWithState` | [`agents.py`](../master/agents.py) |
| 3       | Problem            | `Problem` | [`search.py`](../master/search.py) |
| 3       | Node               | `Node` | [`search.py`](../master/search.py) |
| 3       | Queue              | `Queue` | [`utils.py`](../master/utils.py) |
| 3.1     | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`search.py`](../master/search.py) |
| 3.2     | Romania            | `romania` | [`search.py`](../master/search.py) |
| 3.7     | Tree-Search        | `tree_search` | [`search.py`](../master/search.py) |
| 3.7     | Graph-Search        | `graph_search` | [`search.py`](../master/search.py) |
| 3.11    | Breadth-First-Search        | `breadth_first_search` | [`search.py`](../master/search.py) |
| 3.14    | Uniform-Cost-Search        | `uniform_cost_search` | [`search.py`](../master/search.py) |
| 3.17    | Depth-Limited-Search | `depth_limited_search` | [`search.py`](../master/search.py) |
| 3.18    | Iterative-Deepening-Search | `iterative_deepening_search` | [`search.py`](../master/search.py) |
| 3.22    | Best-First-Search  | `best_first_graph_search` | [`search.py`](../master/search.py) |
| 3.24    | A\*-Search        | `astar_search` | [`search.py`](../master/search.py) |
| 3.26    | Recursive-Best-First-Search | `recursive_best_first_search` | [`search.py`](../master/search.py) |
| 4.2     | Hill-Climbing      | `hill_climbing` | [`search.py`](../master/search.py) |
| 4.5     | Simulated-Annealing | `simulated_annealing` | [`search.py`](../master/search.py) |
| 4.8     | Genetic-Algorithm  | `genetic_algorithm` | [`search.py`](../master/search.py) |
| 4.11    | And-Or-Graph-Search | `and_or_graph_search` | [`search.py`](../master/search.py)  |
| 4.21    | Online-DFS-Agent   | `online_dfs_agent` | [`search.py`](../master/search.py) |
| 4.24    | LRTA\*-Agent       |        |        |
| 5.3     | Minimax-Decision   | `minimax_decision` | [`games.py`](../master/games.py) |
| 5.7     | Alpha-Beta-Search  | `alphabeta_search` | [`games.py`](../master/games.py) |
| 6       | CSP                | `CSP` | [`csp.py`](../master/csp.py) |
| 6.3     | AC-3               | `AC3` | [`csp.py`](../master/csp.py) |
| 6.5     | Backtracking-Search | `backtracking_search` | [`csp.py`](../master/csp.py) |
| 6.8     | Min-Conflicts      | `min_conflicts` | [`csp.py`](../master/csp.py) |
| 6.11    | Tree-CSP-Solver    | `tree_csp_solver` | [`csp.py`](../master/csp.py) |
| 7       | KB                 | `KB` | [`logic.py`](../master/logic.py) |
| 7.1     | KB-Agent           | `KB_Agent` | [`logic.py`](../master/logic.py) |
| 7.7     | Propositional Logic Sentence | `Expr` | [`logic.py`](../master/logic.py) |
| 7.10    | TT-Entails         | `tt_entials` | [`logic.py`](../master/logic.py) |
| 7.12    | PL-Resolution      | `pl_resolution` | [`logic.py`](../master/logic.py) |
| 7.14    | Convert to CNF     | `to_cnf` | [`logic.py`](../master/logic.py) |
| 7.15    | PL-FC-Entails?     | `pl_fc_resolution` | [`logic.py`](../master/logic.py) |
| 7.17    | DPLL-Satisfiable?  | `dpll_satisfiable` | [`logic.py`](../master/logic.py) |
| 7.18    | WalkSAT            | `WalkSAT` | [`logic.py`](../master/logic.py) |
| 7.20    | Hybrid-Wumpus-Agent    |         |           |
| 7.22    | SATPlan            |          |
| 9       | Subst              | `subst` | [`logic.py`](../master/logic.py) |
| 9.1     | Unify              | `unify` | [`logic.py`](../master/logic.py) |
| 9.3     | FOL-FC-Ask         | `fol_fc_ask` | [`logic.py`](../master/logic.py) |
| 9.6     | FOL-BC-Ask         | `fol_bc_ask` | [`logic.py`](../master/logic.py) |
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
| \* 12.6    | House-Building-Problem |          |
| \* 12.22   | Continuous-POP-Agent |          |
| 11.10   | Doubles-tennis     |          |
| 13      | Discrete Probability Distribution | `ProbDist` | [`probability.py`](../master/probability.py) |
| 13.1    | DT-Agent           | `DTAgent` | [`probability.py`](../master/probability.py) |
| \* 13.4    | Enumerate-Joint-Ask | `enumerate_joint_ask` | [`probability.py`](../master/probability.py) |
| 14.9    | Enumeration-Ask    | `enumeration_ask` | [`probability.py`](../master/probability.py) |
| 14.11   | Elimination-Ask    | `elimination_ask` | [`probability.py`](../master/probability.py) |
| 14.13   | Prior-Sample       | `prior_sample` | [`probability.py`](../master/probability.py) |
| 14.14   | Rejection-Sampling | `rejection_sampling` | [`probability.py`](../master/probability.py) |
| 14.15   | Likelihood-Weighting | `likelihood_weighting` | [`probability.py`](../master/probability.py) |
| 14.16   | Gibbs-Ask           |          |
| 15.4    | Forward-Backward   | `forward_backward` | [`probability.py`](../master/probability.py) |
| 15.6    | Fixed-Lag-Smoothing | `fixed_lag_smoothing` | [`probability.py`](../master/probability.py) |
| 15.17   | Particle-Filtering | `particle_filtering` | [`probability.py`](../master/probability.py) |
| 16.9    | Information-Gathering-Agent |          |
| 17.4    | Value-Iteration    | `value_iteration` | [`mdp.py`](../master/mdp.py) |
| 17.7    | Policy-Iteration   | `policy_iteration` | [`mdp.py`](../master/mdp.py) |
| 17.7    | POMDP-Value-Iteration  |           |        |
| 18.5    | Decision-Tree-Learning | `DecisionTreeLearner` | [`learning.py`](../master/learning.py) |
| 18.8    | Cross-Validation   | `cross_validation` | [`learning.py`](../master/learning.py) |
| 18.11   | Decision-List-Learning |          |
| 18.24   | Back-Prop-Learning |          |
| 18.34   | AdaBoost           | `AdaBoost` | [`learning.py`](../master/learning.py) |
| 19.2    | Current-Best-Learning |          |
| 19.3    | Version-Space-Learning |          |
| 19.8    | Minimal-Consistent-Det |          |
| 19.12   | FOIL               |          |
| 21.2    | Passive-ADP-Agent  | `PassiveADPAgent` | [`rl.py`](../master/rl.py) |
| 21.4    | Passive-TD-Agent   | `PassiveTDAgent` | [`rl.py`](../master/rl.py) |
| 21.8    | Q-Learning-Agent   | `QLearningAgent` | [`rl.py`](../master/rl.py) |
| \* 21.2    | Naive-Communicating-Agent |          |
| 22.1    | HITS               |         |         |
| 23      | Chart-Parse        | `Chart` | [`nlp.py`](../master/nlp.py) |
| 23.5    | CYK-Parse          |         |         |
| \* 23.1    | Viterbi-Segmentation | `viterbi_segment` | [`text.py`](../master/text.py) |
| \* 24.21   | Align              |          |
| 25.9    | Monte-Carlo-Localization|       |


# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is on GitHub, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Thanks to all!
