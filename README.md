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

Here is a table of algorithms, the figure and page where they appear in the book, and the file where they appear in the code. Unfortuately, this chart was made for the old second edition; and has only been partially upfdated to third edition, and not at all to fourth edition. We could use help fixing up the table, based on the figures in [algorithms.pdf](https://github.com/aimacode/aima-pseudocode/blob/master/algorithms.pdf). Empty implementations are a good place for contributors to look for an iassue.


| **Fig** | **Page** | **Name (in 3<sup>rd<sup> edition)** | **Name (in code)** | **File**
|:--------|:---------|:-------------------|:---------|:-----------|
| 2.1     |  36      | Environment        | `Environment` | [`agents.py`](../master/agents.py) |
| 2.1     |  36      | Agent              | `Agent` | [`agents.py`](../master/agents.py) |
| 2.3     |  37      | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.7     |  48      | Table-Driven-Agent | `TableDrivenAgent` | [`agents.py`](../master/agents.py) |
| 2.8     |  49      | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.10    |  50      | Simple-Reflex-Agent | `SimpleReflexAgent` | [`agents.py`](../master/agents.py) |
| 2.12    |  52      | Model-Based-Reflex-Agent | `ReflexAgentWithState` | [`agents.py`](../master/agents.py) |
| 3.1     |  69      | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`search.py`](../master/search.py) |
| 3       |  62      | Problem            | `Problem` | [`search.py`](../master/search.py) |
| 3.2     |  70      | Romania            | `romania` | [`search.py`](../master/search.py) |
| 3       |  69      | Node               | `Node` | [`search.py`](../master/search.py) |
| 3       |  71      | Queue              | `Queue` | [`utils.py`](../master/utils.py) |
| 3.7     |  79      | Tree-Search        | `tree_search` | [`search.py`](../master/search.py) |
| 3.7     |  79      | Graph-Search        | `graph_search` | [`search.py`](../master/search.py) |
| 3.11    |  84     | Breadth-First-Search        | `breadth_first_search` | [`search.py`](../master/search.py) |
| 3.14    |  86     | Uniform-Cost-Search        | `uniform_cost_search` | [`search.py`](../master/search.py) |
| 3.17    |  90      | Depth-Limited-Search | `depth_limited_search` | [`search.py`](../master/search.py) |
| 3.18    |  91      | Iterative-Deepening-Search | `iterative_deepening_search` | [`search.py`](../master/search.py) |
| 3.19    |  83      | Graph-Search       | `graph_search` | [`search.py`](../master/search.py) |
| 4       |  95      | Best-First-Search  | `best_first_graph_search` | [`search.py`](../master/search.py) |
| 4       |  97      | A\*-Search        | `astar_search` | [`search.py`](../master/search.py) |
| 3.26    | 101      | Recursive-Best-First-Search | `recursive_best_first_search` | [`search.py`](../master/search.py) |
| 4.2     | 125      | Hill-Climbing      | `hill_climbing` | [`search.py`](../master/search.py) |
| 4.5     | 129      | Simulated-Annealing | `simulated_annealing` | [`search.py`](../master/search.py) |
| 4.8     | 132      | Genetic-Algorithm  | `genetic_algorithm` | [`search.py`](../master/search.py) |
| 4.21    | 153      | Online-DFS-Agent   | `online_dfs_agent` | [`search.py`](../master/search.py) |
| 4.24    | 155      | LRTA\*-Agent      | `lrta_star_agent` | [`search.py`](../master/search.py) |
| 5       | 137      | CSP                | `CSP` | [`csp.py`](../master/csp.py) |
| 5.3     | 169      | Minimax-Decision   | `minimax_decision` | [`games.py`](../master/games.py) |
| 5.7     | 173      | Alpha-Beta-Search  | `alphabeta_search` | [`games.py`](../master/games.py) |
| 7       | 195      | KB                 | `KB` | [`logic.py`](../master/logic.py) |
| 6.1     | 208      | KB-Agent           | `KB_Agent` | [`logic.py`](../master/logic.py) |
| 6.7     | 216      | Propositional Logic Sentence | `Expr` | [`logic.py`](../master/logic.py) |
| 6.10    | 220      | TT-Entails         | `tt_entials` | [`logic.py`](../master/logic.py) |
| 7       | 215      | Convert to CNF     | `to_cnf` | [`logic.py`](../master/logic.py) |
| 6.12    | 227      | PL-Resolution      | `pl_resolution` | [`logic.py`](../master/logic.py) |
| 6.15    | 230      | PL-FC-Entails?     | `pl_fc_resolution` | [`logic.py`](../master/logic.py) |
| 6.17    | 233      | DPLL-Satisfiable?  | `dpll_satisfiable` | [`logic.py`](../master/logic.py) |
| 6.18    | 235      | WalkSAT            | `WalkSAT` | [`logic.py`](../master/logic.py) |
| 6.20    | 242      | Hybrid-Wumpus-Agent    | `HybridWumpusAgent` | [`logic.py`](../master/logic.py) |
| 6.22    | 244      | SATPlan            |          |
| 7.3     | 265      | AC-3               | `AC3` | [`csp.py`](../master/csp.py) |
| 7.5     | 271      | Backtracking-Search | `backtracking_search` | [`csp.py`](../master/csp.py) |
| 7.8     | 277      | Min-Conflicts      | `min_conflicts` | [`csp.py`](../master/csp.py) |
| 9       | 273      | Subst              | `subst` | [`logic.py`](../master/logic.py) |
| 9.1     | 334      | Unify              | `unify` | [`logic.py`](../master/logic.py) |
| 9.3     | 338      | FOL-FC-Ask         | `fol_fc_ask` | [`logic.py`](../master/logic.py) |
| 9.6     | 344      | FOL-BC-Ask         | `fol_bc_ask` | [`logic.py`](../master/logic.py) |
| 9.14    | 307      | Otter              |          |
| 10.1    | 376      | Air-Cargo-problem    |          |
| 10.2    | 377      | Spare-Tire-Problem |          |
| 10.3    | 378      | Three-Block-Tower  |          |
| 11      | 390      | Partial-Order-Planner |          |
| 10.7    | 387      | Cake-Problem       |          |
| 10.9    | 390      | Graphplan          |          |
| 12.1    | 418      | Job-Shop-Problem   |          |
| 11.1    | 409      | Job-Shop-Problem-With-Resources |          |
| 12.6    | 424      | House-Building-Problem |          |
| 12.10   | 435      | And-Or-Graph-Search | `and_or_graph_search` | [`search.py`](../master/search.py)  |
| 12.22   | 449      | Continuous-POP-Agent |          |
| 12.23   | 450      | Doubles-tennis     |          |
| 13.1    | 492      | DT-Agent           | `DTAgent` | [`probability.py`](../master/probability.py) |
| 13      | 469      | Discrete Probability Distribution | `DiscreteProbDist` | [`probability.py`](../master/probability.py) |
| 13.4    | 477      | Enumerate-Joint-Ask | `enumerate_joint_ask` | [`probability.py`](../master/probability.py) |
| 14.11   | 537      | Elimination-Ask    | `elimination_ask` | [`probability.py`](../master/probability.py) |
| 14.13   | 540      | Prior-Sample       | `prior_sample` | [`probability.py`](../master/probability.py) |
| 14.14   | 542      | Rejection-Sampling | `rejection_sampling` | [`probability.py`](../master/probability.py) |
| 14.15   | 543      | Likelihood-Weighting | `likelihood_weighting` | [`probability.py`](../master/probability.py) |
| 14.15   | 517      | MCMC-Ask           |          |
| 15.4    | 586      | Forward-Backward   | `forward_backward` | [`probability.py`](../master/probability.py) |
| 15.6    | 590      | Fixed-Lag-Smoothing | `fixed_lag_smoothing` | [`probability.py`](../master/probability.py) |
| 15.17   | 608      | Particle-Filtering | `particle_filtering` | [`probability.py`](../master/probability.py) |
| 16.9    | 643      | Information-Gathering-Agent |          |
| 17.4    | 664      | Value-Iteration    | `value_iteration` | [`mdp.py`](../master/mdp.py) |
| 17.7    | 668      | Policy-Iteration   | `policy_iteration` | [`mdp.py`](../master/mdp.py) |
| 18.5    | 713     | Decision-Tree-Learning | `DecisionTreeLearner` | [`learning.py`](../master/learning.py) |
| 18.11   | 728      | Decision-List-Learning |          |
| 18.24   | 745      | Back-Prop-Learning |          |
| 18.34   | 762      | AdaBoost           | `AdaBoost` | [`learning.py`](../master/learning.py) |
| 19.2    | 783      | Current-Best-Learning |          |
| 19.3    | 785      | Version-Space-Learning |          |
| 19.8    | 798      | Minimal-Consistent-Det |          |
| 19.12   | 805      | FOIL               |          |
| 20.21   | 742      | Perceptron-Learning | `PerceptronLearner` | [`learning.py`](../master/learning.py) |
| 22.2    | 877      | Passive-ADP-Agent  | `PassiveADPAgent` | [`rl.py`](../master/rl.py) |
| 22.4    | 880      | Passive-TD-Agent   | `PassiveTDAgent` | [`rl.py`](../master/rl.py) |
| 22.8    | 887      | Q-Learning-Agent   |          |
| 22.2    | 796      | Naive-Communicating-Agent |          |
| 22.7    | 801      | Chart-Parse        | `Chart` | [`nlp.py`](../master/nlp.py) |
| 23.1    | 837      | Viterbi-Segmentation | `viterbi_segment` | [`text.py`](../master/text.py) |
| 24.21   | 892      | Align              |          |
| 25.9    | 999      | Monte-Carlo-Localization|       |


# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is in Githib, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Thanks to all!
