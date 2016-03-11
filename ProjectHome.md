Python implementation of algorithms from Russell and Norvig's _[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu)_.

The [Subversion checkout](http://code.google.com/p/aima-python/source/checkout) is actively developed as of October 2011; you'll probably prefer it to the .zip download for now.

# Index of Code #

| **Fig** | **Page** | **Name (in book)** | **Code** |
|:--------|:---------|:-------------------|:---------|
| 2       |  32      | Environment        | [Environment](../master/agents.py) |
| 2.1     |  33      | Agent              | [Agent](../master/agents.py) |
| 2.3     |  34      | Table-Driven-Vacuum-Agent | [TableDrivenVacuumAgent](../master/agents.py) |
| 2.7     |  45      | Table-Driven-Agent | [TableDrivenAgent](../master/agents.py) |
| 2.8     |  46      | Reflex-Vacuum-Agent | [ReflexVacuumAgent](../master/agents.py) |
| 2.10    |  47      | Simple-Reflex-Agent | [SimpleReflexAgent](../master/agents.py) |
| 2.12    |  49      | Reflex-Agent-With-State | [ReflexAgentWithState](../master/agents.py) |
| 3.1     |  61      | Simple-Problem-Solving-Agent | [SimpleProblemSolvingAgent](../master/search.py) |
| 3       |  62      | Problem            | [Problem](../master/search.py) |
| 3.2     |  63      | Romania            | [romania](../master/search.py) |
| 3       |  69      | Node               | [Node](../master/search.py) |
| 3.7     |  70      | Tree-Search        | [tree\_search](../master/search.py) |
| 3       |  71      | Queue              | [Queue](../master/utils.py) |
| 3.9     |  72      | Tree-Search        | [tree\_search](../master/search.py) |
| 3.13    |  77      | Depth-Limited-Search | [depth\_limited\_search](../master/search.py) |
| 3.14    |  79      | Iterative-Deepening-Search | [iterative\_deepening\_search](../master/search.py) |
| 3.19    |  83      | Graph-Search       | [graph\_search](../master/search.py) |
| 4       |  95      | Best-First-Search  | [best\_first\_graph\_search](../master/search.py) |
| 4       |  97      | A`*`-Search        | [astar\_search](../master/search.py) |
| 4.5     | 102      | Recursive-Best-First-Search | [recursive\_best\_first\_search](../master/search.py) |
| 4.11    | 112      | Hill-Climbing      | [hill\_climbing](../master/search.py) |
| 4.14    | 116      | Simulated-Annealing | [simulated\_annealing](../master/search.py) |
| 4.17    | 119      | Genetic-Algorithm  | [genetic\_algorithm](../master/search.py) |
| 4.20    | 126      | Online-DFS-Agent   | [online\_dfs\_agent](../master/search.py) |
| 4.23    | 128      | LRTA`*`-Agent      | [lrta\_star\_agent](../master/search.py) |
| 5       | 137      | CSP                | [CSP](../master/csp.py) |
| 5.3     | 142      | Backtracking-Search | [backtracking\_search](../master/csp.py) |
| 5.7     | 146      | AC-3               | [AC3](../master/csp.py) |
| 5.8     | 151      | Min-Conflicts      | [min\_conflicts](../master/csp.py) |
| 6.3     | 166      | Minimax-Decision   | [minimax\_decision](../master/games.py) |
| 6.7     | 170      | Alpha-Beta-Search  | [alphabeta\_search](../master/games.py) |
| 7       | 195      | KB                 | [KB](../master/logic.py) |
| 7.1     | 196      | KB-Agent           | [KB\_Agent](../master/logic.py) |
| 7.7     | 205      | Propositional Logic Sentence | [Expr](../master/logic.py) |
| 7.10    | 209      | TT-Entails         | [tt\_entials](../master/logic.py) |
| 7       | 215      | Convert to CNF     | [to\_cnf](../master/logic.py) |
| 7.12    | 216      | PL-Resolution      | [pl\_resolution](../master/logic.py) |
| 7.14    | 219      | PL-FC-Entails?     | [pl\_fc\_resolution](../master/logic.py) |
| 7.16    | 222      | DPLL-Satisfiable?  | [dpll\_satisfiable](../master/logic.py) |
| 7.17    | 223      | WalkSAT            | [WalkSAT](../master/logic.py) |
| 7.19    | 226      | PL-Wumpus-Agent    | [PLWumpusAgent](../master/logic.py) |
| 9       | 273      | Subst              | [subst](../master/logic.py) |
| 9.1     | 278      | Unify              | [unify](../master/logic.py) |
| 9.3     | 282      | FOL-FC-Ask         | [fol\_fc\_ask](../master/logic.py) |
| 9.6     | 288      | FOL-BC-Ask         | [fol\_bc\_ask](../master/logic.py) |
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
| 12.10   | 435      | And-Or-Graph-Search | [and\_or\_graph\_search](../master/search.py)  |
| 12.22   | 449      | Continuous-POP-Agent |          |
| 12.23   | 450      | Doubles-tennis     |          |
| 13.1    | 466      | DT-Agent           | [DTAgent](../master/probability.py) |
| 13      | 469      | Discrete Probability Distribution | [DiscreteProbDist](../master/probability.py) |
| 13.4    | 477      | Enumerate-Joint-Ask | [enumerate\_joint\_ask](../master/probability.py) |
| 14.10   | 509      | Elimination-Ask    | [elimination\_ask](../master/probability.py) |
| 14.12   | 512      | Prior-Sample       | [prior\_sample](../master/probability.py) |
| 14.13   | 513      | Rejection-Sampling | [rejection\_sampling](../master/probability.py) |
| 14.14   | 515      | Likelihood-Weighting | [likelihood\_weighting](../master/probability.py) |
| 14.15   | 517      | MCMC-Ask           |          |
| 15.4    | 546      | Forward-Backward   | [forward\_backward](../master/probability.py) |
| 15.6    | 552      | Fixed-Lag-Smoothing | [fixed\_lag\_smoothing](../master/probability.py) |
| 15.15   | 566      | Particle-Filtering | [particle\_filtering](../master/probability.py) |
| 16.8    | 603      | Information-Gathering-Agent |          |
| 17.4    | 621      | Value-Iteration    | [value\_iteration](../master/mdp.py) |
| 17.7    | 624      | Policy-Iteration   | [policy\_iteration](../master/mdp.py) |
| 18.5    | 658      | Decision-Tree-Learning | [DecisionTreeLearner](../master/learning.py) |
| 18.10   | 667      | AdaBoost           | [AdaBoost](../master/learning.py) |
| 18.14   | 672      | Decision-List-Learning |          |
| 19.2    | 681      | Current-Best-Learning |          |
| 19.3    | 683      | Version-Space-Learning |          |
| 19.8    | 696      | Minimal-Consistent-Det |          |
| 19.12   | 702      | FOIL               |          |
| 20.21   | 742      | Perceptron-Learning | [PerceptronLearner](../master/learning.py) |
| 20.25   | 746      | Back-Prop-Learning |          |
| 21.2    | 768      | Passive-ADP-Agent  | [PassiveADPAgent](../master/rl.py) |
| 21.4    | 769      | Passive-TD-Agent   | [PassiveTDAgent](../master/rl.py) |
| 21.8    | 776      | Q-Learning-Agent   |          |
| 22.2    | 796      | Naive-Communicating-Agent |          |
| 22.7    | 801      | Chart-Parse        | [Chart](../master/nlp.py) |
| 23.1    | 837      | Viterbi-Segmentation | [viterbi\_segment](../master/text.py) |
| 24.21   | 892      | Align              |          |
