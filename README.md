<div align="center">
  <a href="http://aima.cs.berkeley.edu/"><img src="https://raw.githubusercontent.com/aimacode/aima-python/master/images/aima_logo.png"></a><br><br>
</div>

# `aima-python` [![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python) [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/aimacode/aima-python)


Python code for the book *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu).* You can use this in conjunction with a course on AI, or for study on your own. We're looking for [solid contributors](https://github.com/aimacode/aima-python/blob/master/CONTRIBUTING.md) to help.



## Structure of the Project

When complete, this project will have Python implementations for all the pseudocode algorithms in the book, as well as tests and examples of use. For each major topic, such as `nlp` (natural language processing), we provide the following  files:

- `nlp.py`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data.
- `tests/test_nlp.py`: A lightweight test suite, using `assert` statements, designed for use with [`py.test`](http://pytest.org/latest/), but also usable on their own.
- `nlp.ipynb`: A Jupyter (IPython) notebook that explains and gives examples of how to use the code.
- `nlp_apps.ipynb`: A Jupyter notebook that gives example applications of the code.


## Python 3.4 and up

This code requires Python 3.4 or later, and does not run in Python 2. You can [install Python](https://www.python.org/downloads) or use a browser-based Python interpreter such as [repl.it](https://repl.it/languages/python3).
You can run the code in an IDE, or from the command line with `python -i filename.py` where the `-i` option puts you in an interactive loop where you can run Python functions. See [jupyter.org](http://jupyter.org/) for instructions on setting up your own Jupyter notebook environment, or run the notebooks online with [try.jupiter.org](https://try.jupyter.org/). 

# Index of Algorithms

Here is a table of algorithms, the figure, name of the algorithm in the book and in the repository, and the file where they are implemented in the repository. This chart was made for the third edition of the book and is being updated for the upcoming fourth edition. Empty implementations are a good place for contributors to look for an issue. The [aima-pseudocode](https://github.com/aimacode/aima-pseudocode) project describes all the algorithms from the book. An asterisk next to the file name denotes the algorithm is not fully implemented. Another great place for contributors to start is by adding tests and writing on the notebooks. You can see which algorithms have tests and notebook sections below. If the algorithm you want to work on is covered, don't worry! You can still add more tests and provide some examples of use in the notebook!

| **Figure** | **Name (in 3<sup>rd</sup> edition)** | **Name (in repository)** | **File** | **Tests** | **Notebook**
|:-------|:----------------------------------|:------------------------------|:--------------------------------|:-----|:---------|
| 2.1    | Environment                       | `Environment`                 | [`agents.py`][agents]           | Done | Included |
| 2.1    | Agent                             | `Agent`                       | [`agents.py`][agents]           | Done | Included |
| 2.3    | Table-Driven-Vacuum-Agent         | `TableDrivenVacuumAgent`      | [`agents.py`][agents]           |      |          |
| 2.7    | Table-Driven-Agent                | `TableDrivenAgent`            | [`agents.py`][agents]           |      |          |
| 2.8    | Reflex-Vacuum-Agent               | `ReflexVacuumAgent`           | [`agents.py`][agents]           | Done |          |
| 2.10   | Simple-Reflex-Agent               | `SimpleReflexAgent`           | [`agents.py`][agents]           |      |          |
| 2.12   | Model-Based-Reflex-Agent          | `ReflexAgentWithState`        | [`agents.py`][agents]           |      |          |
| 3      | Problem                           | `Problem`                     | [`search.py`][search]           | Done |          |
| 3      | Node                              | `Node`                        | [`search.py`][search]           | Done |          |
| 3      | Queue                             | `Queue`                       | [`utils.py`][utils]             | Done |          |
| 3.1    | Simple-Problem-Solving-Agent      | `SimpleProblemSolvingAgent`   | [`search.py`][search]           |      |          |
| 3.2    | Romania                           | `romania`                     | [`search.py`][search]           | Done | Included |
| 3.7    | Tree-Search                       | `tree_search`                 | [`search.py`][search]           | Done |          |
| 3.7    | Graph-Search                      | `graph_search`                | [`search.py`][search]           | Done |          |
| 3.11   | Breadth-First-Search              | `breadth_first_search`        | [`search.py`][search]           | Done | Included |
| 3.14   | Uniform-Cost-Search               | `uniform_cost_search`         | [`search.py`][search]           | Done | Included |
| 3.17   | Depth-Limited-Search              | `depth_limited_search`        | [`search.py`][search]           | Done |          |
| 3.18   | Iterative-Deepening-Search        | `iterative_deepening_search`  | [`search.py`][search]           | Done |          |
| 3.22   | Best-First-Search                 | `best_first_graph_search`     | [`search.py`][search]           | Done |          |
| 3.24   | A\*-Search                        | `astar_search`                | [`search.py`][search]           | Done | Included |
| 3.26   | Recursive-Best-First-Search       | `recursive_best_first_search` | [`search.py`][search]           | Done |          |
| 4.2    | Hill-Climbing                     | `hill_climbing`               | [`search.py`][search]           | Done |          |
| 4.5    | Simulated-Annealing               | `simulated_annealing`         | [`search.py`][search]           | Done |          |
| 4.8    | Genetic-Algorithm                 | `genetic_algorithm`           | [`search.py`][search]           | Done | Included |
| 4.11   | And-Or-Graph-Search               | `and_or_graph_search`         | [`search.py`][search]           | Done |          |
| 4.21   | Online-DFS-Agent                  | `online_dfs_agent`            | [`search.py`][search]           |      |          |
| 4.24   | LRTA\*-Agent                      | `LRTAStarAgent`               | [`search.py`][search]           | Done |          |
| 5.3    | Minimax-Decision                  | `minimax_decision`            | [`games.py`][games]             | Done | Included |
| 5.7    | Alpha-Beta-Search                 | `alphabeta_search`            | [`games.py`][games]             | Done | Included |
| 6      | CSP                               | `CSP`                         | [`csp.py`][csp]                 | Done | Included |
| 6.3    | AC-3                              | `AC3`                         | [`csp.py`][csp]                 | Done |          |
| 6.5    | Backtracking-Search               | `backtracking_search`         | [`csp.py`][csp]                 | Done | Included |
| 6.8    | Min-Conflicts                     | `min_conflicts`               | [`csp.py`][csp]                 | Done |          |
| 6.11   | Tree-CSP-Solver                   | `tree_csp_solver`             | [`csp.py`][csp]                 | Done | Included |
| 7      | KB                                | `KB`                          | [`logic.py`][logic]             | Done | Included |
| 7.1    | KB-Agent                          | `KB_Agent`                    | [`logic.py`][logic]             | Done |          |
| 7.7    | Propositional Logic Sentence      | `Expr`                        | [`logic.py`][logic]             | Done |          |
| 7.10   | TT-Entails                        | `tt_entails`                  | [`logic.py`][logic]             | Done |          |
| 7.12   | PL-Resolution                     | `pl_resolution`               | [`logic.py`][logic]             | Done | Included |
| 7.14   | Convert to CNF                    | `to_cnf`                      | [`logic.py`][logic]             | Done |          |
| 7.15   | PL-FC-Entails?                    | `pl_fc_resolution`            | [`logic.py`][logic]             | Done |          |
| 7.17   | DPLL-Satisfiable?                 | `dpll_satisfiable`            | [`logic.py`][logic]             | Done |          |
| 7.18   | WalkSAT                           | `WalkSAT`                     | [`logic.py`][logic]             | Done |          |
| 7.20   | Hybrid-Wumpus-Agent               | `HybridWumpusAgent`           |                                 |      |          |
| 7.22   | SATPlan                           | `SAT_plan`                    | [`logic.py`][logic]             | Done |          |
| 9      | Subst                             | `subst`                       | [`logic.py`][logic]             | Done |          |
| 9.1    | Unify                             | `unify`                       | [`logic.py`][logic]             | Done | Included |
| 9.3    | FOL-FC-Ask                        | `fol_fc_ask`                  | [`logic.py`][logic]             | Done |          |
| 9.6    | FOL-BC-Ask                        | `fol_bc_ask`                  | [`logic.py`][logic]             | Done |          |
| 9.8    | Append                            |                               |                                 |      |          |
| 10.1   | Air-Cargo-problem                 | `air_cargo`                   | [`planning.py`][planning]       | Done |          |
| 10.2   | Spare-Tire-Problem                | `spare_tire`                  | [`planning.py`][planning]       | Done |          |
| 10.3   | Three-Block-Tower                 | `three_block_tower`           | [`planning.py`][planning]       | Done |          |
| 10.7   | Cake-Problem                      | `have_cake_and_eat_cake_too`  | [`planning.py`][planning]       | Done |          |
| 10.9   | Graphplan                         | `GraphPlan`                   | [`planning.py`][planning]       |      |          |
| 10.13  | Partial-Order-Planner             |                               |                                 |      |          |
| 11.1   | Job-Shop-Problem-With-Resources   | `job_shop_problem`            | [`planning.py`][planning]       | Done |          |
| 11.5   | Hierarchical-Search               | `hierarchical_search`         | [`planning.py`][planning]       |      |          |
| 11.8   | Angelic-Search                    |                               |                                 |      |          |
| 11.10  | Doubles-tennis                    | `double_tennis_problem`       | [`planning.py`][planning]       |      |          |
| 13     | Discrete Probability Distribution | `ProbDist`                    | [`probability.py`][probability] | Done | Included |
| 13.1   | DT-Agent                          | `DTAgent`                     | [`probability.py`][probability] |      |          |
| 14.9   | Enumeration-Ask                   | `enumeration_ask`             | [`probability.py`][probability] | Done | Included |
| 14.11  | Elimination-Ask                   | `elimination_ask`             | [`probability.py`][probability] | Done | Included |
| 14.13  | Prior-Sample                      | `prior_sample`                | [`probability.py`][probability] |      | Included |
| 14.14  | Rejection-Sampling                | `rejection_sampling`          | [`probability.py`][probability] | Done | Included |
| 14.15  | Likelihood-Weighting              | `likelihood_weighting`        | [`probability.py`][probability] | Done | Included |
| 14.16  | Gibbs-Ask                         | `gibbs_ask`                   | [`probability.py`][probability] |      | Included |
| 15.4   | Forward-Backward                  | `forward_backward`            | [`probability.py`][probability] | Done |          |
| 15.6   | Fixed-Lag-Smoothing               | `fixed_lag_smoothing`         | [`probability.py`][probability] | Done |          |
| 15.17  | Particle-Filtering                | `particle_filtering`          | [`probability.py`][probability] | Done |          |
| 16.9   | Information-Gathering-Agent       |                               |                                 |      |          |
| 17.4   | Value-Iteration                   | `value_iteration`             | [`mdp.py`][mdp]                 | Done | Included |
| 17.7   | Policy-Iteration                  | `policy_iteration`            | [`mdp.py`][mdp]                 | Done |          |
| 17.9   | POMDP-Value-Iteration             |                               |                                 |      |          |
| 18.5   | Decision-Tree-Learning            | `DecisionTreeLearner`         | [`learning.py`][learning]       | Done | Included |
| 18.8   | Cross-Validation                  | `cross_validation`            | [`learning.py`][learning]       |      |          |
| 18.11  | Decision-List-Learning            | `DecisionListLearner`         | [`learning.py`][learning]\*     |      |          |
| 18.24  | Back-Prop-Learning                | `BackPropagationLearner`      | [`learning.py`][learning]       | Done | Included |
| 18.34  | AdaBoost                          | `AdaBoost`                    | [`learning.py`][learning]       |      |          |
| 19.2   | Current-Best-Learning             | `current_best_learning`       | [`knowledge.py`](knowledge.py)  | Done | Included |
| 19.3   | Version-Space-Learning            | `version_space_learning`      | [`knowledge.py`](knowledge.py)  | Done | Included |
| 19.8   | Minimal-Consistent-Det            | `minimal_consistent_det`      | [`knowledge.py`](knowledge.py)  | Done |          |
| 19.12  | FOIL                              | `FOIL_container`              | [`knowledge.py`](knowledge.py)  | Done |          |
| 21.2   | Passive-ADP-Agent                 | `PassiveADPAgent`             | [`rl.py`][rl]                   | Done |          |
| 21.4   | Passive-TD-Agent                  | `PassiveTDAgent`              | [`rl.py`][rl]                   | Done | Included |
| 21.8   | Q-Learning-Agent                  | `QLearningAgent`              | [`rl.py`][rl]                   | Done | Included |
| 22.1   | HITS                              | `HITS`                        | [`nlp.py`][nlp]                 | Done | Included |
| 23     | Chart-Parse                       | `Chart`                       | [`nlp.py`][nlp]                 | Done | Included |
| 23.5   | CYK-Parse                         | `CYK_parse`                   | [`nlp.py`][nlp]                 | Done | Included |
| 25.9   | Monte-Carlo-Localization          | `monte_carlo_localization`    | [`probability.py`][probability] | Done |          |


# Index of data structures

Here is a table of the implemented data structures, the figure, name of the implementation in the repository, and the file where they are implemented.

| **Figure** | **Name (in repository)** | **File** |
|:-------|:--------------------------------|:--------------------------|
| 3.2    | romania_map                     | [`search.py`][search]     |
| 4.9    | vacumm_world                    | [`search.py`][search]     |
| 4.23   | one_dim_state_space             | [`search.py`][search]     |
| 6.1    | australia_map                   | [`search.py`][search]     |
| 7.13   | wumpus_world_inference          | [`logic.py`][logic]       |
| 7.16   | horn_clauses_KB                 | [`logic.py`][logic]       |
| 17.1   | sequential_decision_environment | [`mdp.py`][mdp]           |
| 18.2   | waiting_decision_tree           | [`learning.py`][learning] |


# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is on GitHub, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Many thanks to all contributors, especially @darius, @SnShine, @reachtarunhere, @MrDupin, and @Chipe1.

<!---Reference Links-->
[agents]:../master/agents.py
[csp]:../master/csp.py
[games]:../master/games.py
[grid]:../master/grid.py
[knowledge]:../master/knowledge.py
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
