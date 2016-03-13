# ![](https://github.com/aimacode/aima-java/blob/gh-pages/aima3e/images/aima3e.jpg)`aima-python`[![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python)


Python code for the book *Artificial Intelligence: A Modern Approach.* We're loooking for one student sponsored by Google Summer of Code ([GSoC](https://summerofcode.withgoogle.com/)) to work on this project; if you want to be that student, make some good contributions here (by looking through the [Issues](https://github.com/aimacode/aima-python/issues) and resolving some), and submit an [application](https://summerofcode.withgoogle.com/terms/student). (However, be warned that we've had over 150 students express interest, so competition will be tough.) And we're always looking for solid contributors who are not affiliated with GSoC. A big thank you to everyone who has contributed!

## Python 3.4

This code is in Python 3.4. (Of course, the current version, Python 3.5, also works.) You can [install the latest Python version](https://www.python.org/downloads), and if that doesn't work, use a browser-based Python interpreter such as [repl.it](https://repl.it/languages/python3).

## Structure of the Project

When complete, this project will have Python code for all the pseudocode algorithms in the book. For each major topic, such as `logic`, we will have the following three files in the main branch:

- `logic.py`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data.
- `logic.ipynb`: A Jupyter notebook, with examples of usage. Does a `from logic import *` to get the code.
- `tests/logic_test.py`: A lightweight test suite, using `assert` statements, designed for use with [`py.test`](http://pytest.org/latest/).

Until we get there, we will support a legacy branch, `aima3python2` (for the third edition of the textbook and for Python 2 code). To prepare code for the new master branch, the following two steps should be taken:

## Port to Python 3; Pythonic Idioms; py.test

- Check for common problems in [porting to Python 3](http://python3porting.com/problems.html), such as: `print` is now a function; `range` and `map` and other functions no longer produce `list`s; objects of different types can no longer be compared with `<`; strings are now Unicode; it would be nice to move `%` string formating to `.format`; there is a new `next` function for generators; integer division now returns a float; we can now use set literals.
- Replace old Lisp-based idioms with proper Python idioms. For example, we have many functions that were taken directly from Common Lisp, such as the `every` function: `every(callable, items)` returns true if every element of `items` is callable. This is good Lisp style, but good Python style would be to use `all` and a generator expression: `all(callable(f) for f in items)`. Eventually, fix all calls to these legacy Lisp functions and then remove the functions.
- Add more tests in `_test.py` files. Strive for terseness; it is ok to group multiple asserts into one `def test_something():` function. Move most tests to `_test.py`, but it is fine to have a single `doctest` example in the docstring of a function in the `.py` file, if the purpose of the doctest is to explain how to use the function, rather than test the implementation.

## New and Improved Algorithms

- Implement functions that were in the third edition of the book but were not yet implemented in the code. Check the [list of pseudocode algorithms (pdf)](https://github.com/aimacode/pseudocode/blob/master/algorithms.pdf) to see what's missing.
- As we finish chapters for the new fourth edition, we will share the new pseudocode in the [`aima-pseudocode`](https://github.com/aimacode/aima-pseudocode) repository, and describe what changes are necessary.
We hope to have a `algorithm-name.md` file for each algorithm, eventually; it would be great if contributors could add some for the existing algorithms.
- Give examples of how to use the code in the `.ipynb` file.

# Style Guide

There are a few style rules that are unique to this project:

- The first rule is that the code should correspond directly to the pseudocode in the book. When possible this will be almost one-to-one, just allowing for the syntactic differences between Python and pseudocode, and for different library functions.
- Don't make a function more complicated than the pseudocode in the book, even if the complication would add a nice feature, or give an efficiency gain. Instead, remain faithful to the pseudocode, and if you must, add a new function (not in the book) with the added feature.
- I use functional programming (functions with no side effects) in many cases, but not exclusively (sometimes classes and/or functions with side effects are used). Let the book's pseudocode be the guide.

Beyond the above rules, we use [Pep 8](https://www.python.org/dev/peps/pep-0008), with a few minor exceptions:

- I have set `--max-line-length 100`, not 79.
- You don't need two spaces after a sentence-ending period.
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings; I don't follow [Pep 257](https://www.python.org/dev/peps/pep-0257/).
- Not all constants have to be UPPERCASE.
- At some point I may add [Pep 484](https://www.python.org/dev/peps/pep-0484/) type annotations, but I think I'll hold off for now;
  I want to get more experience with them, and some people may still be in Python 3.4.

# Index of Code #

Here is a table of algorithms, the figure and page where they appear in the book, and the file where they appear in the code. Unfortuately, this chart was made for the old second edition; and has only been partially upfdated to third edition, and not at all to fourth edition. We could use help fixing up the table, based on the figures in [algorithms.pdf](https://github.com/aimacode/aima-pseudocode/blob/master/algorithms.pdf). Empty implementations are a good place for contributors to look for an iassue.


| **Fig** | **Page** | **Name (in book)** | **Name (in code)** | **File**
|:--------|:---------|:-------------------|:---------|:-----------|
| 2       |  32      | Environment        | `Environment` | [`agents.py`](../master/agents.py) |
| 2.1     |  33      | Agent              | `Agent` | [`agents.py`](../master/agents.py) |
| 2.3     |  34      | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.7     |  45      | Table-Driven-Agent | `TableDrivenAgent` | [`agents.py`](../master/agents.py) |
| 2.8     |  46      | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`agents.py`](../master/agents.py) |
| 2.10    |  47      | Simple-Reflex-Agent | `SimpleReflexAgent` | [`agents.py`](../master/agents.py) |
| 2.12    |  49      | Model-Based-Reflex-Agent | `ReflexAgentWithState` | [`agents.py`](../master/agents.py) |
| 3.1     |  61      | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`search.py`](../master/search.py) |
| 3       |  62      | Problem            | `Problem` | [`search.py`](../master/search.py) |
| 3.2     |  63      | Romania            | `romania` | [`search.py`](../master/search.py) |
| 3       |  69      | Node               | `Node` | [`search.py`](../master/search.py) |
| 3       |  71      | Queue              | `Queue` | [`utils.py`](../master/utils.py) |
| 3.7     |  70      | Tree-Search        | `tree_search` | [`search.py`](../master/search.py) |
| 3.7     |  72      | Graph-Search        | `graph_search` | [`search.py`](../master/search.py) |
| 3.11     |  72     | Breadth-First-Search        | `breadth_first_search` | [`search.py`](../master/search.py) |
| 3.13     |  72     | Uniform-Cost-Search        | `uniform_cost_search` | [`search.py`](../master/search.py) |
| 3.16    |  77      | Depth-Limited-Search | `depth_limited_search` | [`search.py`](../master/search.py) |
| 3.14    |  79      | Iterative-Deepening-Search | `iterative_deepening_search` | [`search.py`](../master/search.py) |
| 3.19    |  83      | Graph-Search       | `graph_search` | [`search.py`](../master/search.py) |
| 4       |  95      | Best-First-Search  | `best_first_graph_search` | [`search.py`](../master/search.py) |
| 4       |  97      | A\*-Search        | `astar_search` | [`search.py`](../master/search.py) |
| 4.5     | 102      | Recursive-Best-First-Search | `recursive_best_first_search` | [`search.py`](../master/search.py) |
| 4.11    | 112      | Hill-Climbing      | `hill_climbing` | [`search.py`](../master/search.py) |
| 4.14    | 116      | Simulated-Annealing | `simulated_annealing` | [`search.py`](../master/search.py) |
| 4.17    | 119      | Genetic-Algorithm  | `genetic_algorithm` | [`search.py`](../master/search.py) |
| 4.20    | 126      | Online-DFS-Agent   | `online_dfs_agent` | [`search.py`](../master/search.py) |
| 4.23    | 128      | LRTA\*-Agent      | `lrta_star_agent` | [`search.py`](../master/search.py) |
| 5       | 137      | CSP                | `CSP` | [`csp.py`](../master/csp.py) |
| 5.3     | 142      | Backtracking-Search | `backtracking_search` | [`csp.py`](../master/csp.py) |
| 5.7     | 146      | AC-3               | `AC3` | [`csp.py`](../master/csp.py) |
| 5.8     | 151      | Min-Conflicts      | `min_conflicts` | [`csp.py`](../master/csp.py) |
| 6.3     | 166      | Minimax-Decision   | `minimax_decision` | [`games.py`](../master/games.py) |
| 6.7     | 170      | Alpha-Beta-Search  | `alphabeta_search` | [`games.py`](../master/games.py) |
| 7       | 195      | KB                 | `KB` | [`logic.py`](../master/logic.py) |
| 7.1     | 196      | KB-Agent           | `KB_Agent` | [`logic.py`](../master/logic.py) |
| 7.7     | 205      | Propositional Logic Sentence | `Expr` | [`logic.py`](../master/logic.py) |
| 7.10    | 209      | TT-Entails         | `tt_entials` | [`logic.py`](../master/logic.py) |
| 7       | 215      | Convert to CNF     | `to_cnf` | [`logic.py`](../master/logic.py) |
| 7.12    | 216      | PL-Resolution      | `pl_resolution` | [`logic.py`](../master/logic.py) |
| 7.14    | 219      | PL-FC-Entails?     | `pl_fc_resolution` | [`logic.py`](../master/logic.py) |
| 7.16    | 222      | DPLL-Satisfiable?  | `dpll_satisfiable` | [`logic.py`](../master/logic.py) |
| 7.17    | 223      | WalkSAT            | `WalkSAT` | [`logic.py`](../master/logic.py) |
| 7.19    | 226      | PL-Wumpus-Agent    | `PLWumpusAgent` | [`logic.py`](../master/logic.py) |
| 9       | 273      | Subst              | `subst` | [`logic.py`](../master/logic.py) |
| 9.1     | 278      | Unify              | `unify` | [`logic.py`](../master/logic.py) |
| 9.3     | 282      | FOL-FC-Ask         | `fol_fc_ask` | [`logic.py`](../master/logic.py) |
| 9.6     | 288      | FOL-BC-Ask         | `fol_bc_ask` | [`logic.py`](../master/logic.py) |
| 9.14    | 307      | Otter              |          |
| 11.2    | 380      | Airport-problem    |          |
| 11.3    | 381      | Spare-Tire-Problem |          |
| 11.4    | 383      | Three-Block-Tower  |          |
| 11      | 390      | Partial-Order-Planner |          |
| 11.11   | 396      | Cake-Problem       |          |
| 11.13   | 399      | Graphplan          |          |
| 11.15   | 403      | SATPlan            |          |
| 12.1    | 418      | Job-Shop-Problem   |          |
| 12.3    | 421      | Job-Shop-Problem-With-Resources |          |
| 12.6    | 424      | House-Building-Problem |          |
| 12.10   | 435      | And-Or-Graph-Search | `and_or_graph_search` | [`search.py`](../master/search.py)  |
| 12.22   | 449      | Continuous-POP-Agent |          |
| 12.23   | 450      | Doubles-tennis     |          |
| 13.1    | 466      | DT-Agent           | `DTAgent` | [`probability.py`](../master/probability.py) |
| 13      | 469      | Discrete Probability Distribution | `DiscreteProbDist` | [`probability.py`](../master/probability.py) |
| 13.4    | 477      | Enumerate-Joint-Ask | `enumerate_joint_ask` | [`probability.py`](../master/probability.py) |
| 14.10   | 509      | Elimination-Ask    | `elimination_ask` | [`probability.py`](../master/probability.py) |
| 14.12   | 512      | Prior-Sample       | `prior_sample` | [`probability.py`](../master/probability.py) |
| 14.13   | 513      | Rejection-Sampling | `rejection_sampling` | [`probability.py`](../master/probability.py) |
| 14.14   | 515      | Likelihood-Weighting | `likelihood_weighting` | [`probability.py`](../master/probability.py) |
| 14.15   | 517      | MCMC-Ask           |          |
| 15.4    | 546      | Forward-Backward   | `forward_backward` | [`probability.py`](../master/probability.py) |
| 15.6    | 552      | Fixed-Lag-Smoothing | `fixed_lag_smoothing` | [`probability.py`](../master/probability.py) |
| 15.15   | 566      | Particle-Filtering | `particle_filtering` | [`probability.py`](../master/probability.py) |
| 16.8    | 603      | Information-Gathering-Agent |          |
| 17.4    | 621      | Value-Iteration    | `value_iteration` | [`mdp.py`](../master/mdp.py) |
| 17.7    | 624      | Policy-Iteration   | `policy_iteration` | [`mdp.py`](../master/mdp.py) |
| 18.5    | 658      | Decision-Tree-Learning | `DecisionTreeLearner` | [`learning.py`](../master/learning.py) |
| 18.10   | 667      | AdaBoost           | `AdaBoost` | [`learning.py`](../master/learning.py) |
| 18.14   | 672      | Decision-List-Learning |          |
| 19.2    | 681      | Current-Best-Learning |          |
| 19.3    | 683      | Version-Space-Learning |          |
| 19.8    | 696      | Minimal-Consistent-Det |          |
| 19.12   | 702      | FOIL               |          |
| 20.21   | 742      | Perceptron-Learning | `PerceptronLearner` | [`learning.py`](../master/learning.py) |
| 20.25   | 746      | Back-Prop-Learning |          |
| 21.2    | 768      | Passive-ADP-Agent  | `PassiveADPAgent` | [`rl.py`](../master/rl.py) |
| 21.4    | 769      | Passive-TD-Agent   | `PassiveTDAgent` | [`rl.py`](../master/rl.py) |
| 21.8    | 776      | Q-Learning-Agent   |          |
| 22.2    | 796      | Naive-Communicating-Agent |          |
| 22.7    | 801      | Chart-Parse        | `Chart` | [`nlp.py`](../master/nlp.py) |
| 23.1    | 837      | Viterbi-Segmentation | `viterbi_segment` | [`text.py`](../master/text.py) |
| 24.21   | 892      | Align              |          |

# Choice of Programming Languages

Are we right to concentrate on Java and Python versions of the code? I think so; both languages are popular; Java is
fast enough for our purposes, and has reasonable type declarations (but can be verbose); Python is popular and has a very direct mapping to the pseudocode in the book (but lacks type declarations and can be slow). The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top five most popular languages are:

        Java, C, C++, C#, Python

So it might be reasonable to also support C++/C# at some point in the future. It might also be reasonable to support a language that combines the terse readability of Python with the type safety and speed of Java; perhaps Go or Julia. And finally, Javascript is the language of the browser; it would be nice to have code that runs in the browser, in Javascript or a variant such as Typescript.

There is also a `aima-lisp` project; in 1995 when we wrote the first edition of the book, Lisp was the right choice, but today it is less popular.

What languages are instructors recommending for their AI class? To get an approximate idea, I gave the query <tt>[norvig russell "Modern Approach"](https://www.google.com/webhp#q=russell%20norvig%20%22modern%20approach%22%20java)</tt> along with the names of various languages and looked at the estimated counts of results on
various dates. However, I don't have much confidence in these figures...

|Language  |2004  |2005  |2007  |2010  |2016  |
|--------  |----: |----: |----: |----: |----: |
|[none](http://www.google.com/search?q=norvig+russell+%22Modern+Approach%22)|8,080|20,100|75,200|150,000|132,000|
|[java](http://www.google.com/search?q=java+norvig+russell+%22Modern+Approach%22)|1,990|4,930|44,200|37,000|50,000|
|[c++](http://www.google.com/search?q=c%2B%2B+norvig+russell+%22Modern+Approach%22)|875|1,820|35,300|105,000|35,000|
|[lisp](http://www.google.com/search?q=lisp+norvig+russell+%22Modern+Approach%22)|844|974|30,100|19,000|14,000|
|[prolog](http://www.google.com/search?q=prolog+norvig+russell+%22Modern+Approach%22)|789|2,010|23,200|17,000|16,000|
|[python](http://www.google.com/search?q=python+norvig+russell+%22Modern+Approach%22)|785|1,240|18,400|11,000|12,000|

# Acknowledgements

Many thanks for contributions over the years. I got bug reports, corrected code, and other support from Darius Bacon, Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others. Now that the project is in Githib, you can see the [contributors](https://github.com/aimacode/aima-python/graphs/contributors) who are doing a great job of actively improving the project. Thanks to all!
