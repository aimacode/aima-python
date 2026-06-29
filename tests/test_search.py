import pytest
from aima.search import *
from aima.logic import WumpusPosition

random.seed("aima-python")

romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
vacuum_world = GraphProblemStochastic('State_1', ['State_7', 'State_8'], vacuum_world)
LRTA_problem = OnlineSearchProblem('State_3', 'State_5', one_dim_state_space)
eight_puzzle = EightPuzzle((1, 2, 3, 4, 5, 7, 8, 6, 0))
eight_puzzle2 = EightPuzzle((1, 0, 6, 8, 7, 5, 4, 2), (0, 1, 2, 3, 4, 5, 6, 7, 8))
n_queens = NQueensProblem(8)


def test_find_min_edge():
    assert romania_problem.find_min_edge() == 70


def test_breadth_first_tree_search():
    assert breadth_first_tree_search(
        romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']
    assert breadth_first_graph_search(n_queens).solution() == [0, 4, 7, 5, 2, 6, 1, 3]


def test_breadth_first_graph_search():
    assert breadth_first_graph_search(romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']


def test_best_first_graph_search():
    # uniform_cost_search and astar_search test it indirectly
    assert best_first_graph_search(
        romania_problem,
        lambda node: node.state).solution() == ['Sibiu', 'Fagaras', 'Bucharest']
    assert best_first_graph_search(
        romania_problem,
        lambda node: node.state[::-1]).solution() == ['Timisoara',
                                                      'Lugoj',
                                                      'Mehadia',
                                                      'Drobeta',
                                                      'Craiova',
                                                      'Pitesti',
                                                      'Bucharest']


def test_uniform_cost_search():
    assert uniform_cost_search(
        romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert uniform_cost_search(n_queens).solution() == [0, 4, 7, 5, 2, 6, 1, 3]


def test_depth_first_tree_search():
    assert depth_first_tree_search(n_queens).solution() == [7, 3, 0, 2, 5, 1, 6, 4]


def test_depth_first_graph_search():
    solution = depth_first_graph_search(romania_problem).solution()
    assert solution[-1] == 'Bucharest'


def test_iterative_deepening_search():
    assert iterative_deepening_search(
        romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']


def test_depth_limited_search():
    solution_3 = depth_limited_search(romania_problem, 3).solution()
    assert solution_3[-1] == 'Bucharest'
    assert depth_limited_search(romania_problem, 2) == 'cutoff'
    solution_50 = depth_limited_search(romania_problem).solution()
    assert solution_50[-1] == 'Bucharest'


def test_bidirectional_search():
    assert bidirectional_search(romania_problem) == 418
    assert bidirectional_search(eight_puzzle) == 12
    assert bidirectional_search(EightPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))) == 2


def test_astar_search():
    assert astar_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert astar_search(eight_puzzle).solution() == ['LEFT', 'LEFT', 'UP', 'RIGHT', 'RIGHT', 'DOWN', 'LEFT', 'UP',
                                                     'LEFT', 'DOWN', 'RIGHT', 'RIGHT']
    assert astar_search(EightPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))).solution() == ['RIGHT', 'RIGHT']
    assert astar_search(n_queens).solution() == [7, 1, 3, 0, 6, 4, 2, 5]


def test_tree_search_variants():
    # n-queens is tree-structured (one queen per column, no repeated states), so
    # the tree variants explore exactly like the graph ones and find the same solution
    assert astar_tree_search(n_queens).solution() == astar_search(n_queens).solution()
    # on Romania (a cyclic graph) cost-bounded tree search still terminates and
    # returns the optimal path, matching the graph versions
    assert uniform_cost_tree_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert astar_tree_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert astar_tree_search(romania_problem).path_cost == 418
    # the greedy tree-search alias mirrors greedy_best_first_graph_search
    assert greedy_best_first_tree_search is best_first_tree_search


def test_iterative_deepening_astar_search():
    # IDA* is optimal, so it returns a solution of the same cost as A*.
    assert iterative_deepening_astar_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert (iterative_deepening_astar_search(eight_puzzle).path_cost ==
            astar_search(eight_puzzle).path_cost)
    assert iterative_deepening_astar_search(EightPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))).solution() == ['RIGHT', 'RIGHT']


def test_traveling_salesman():
    cities = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (0.5, 2)}
    tsp = TravelingSalesman(cities, initial=(0,))
    solution = astar_search(tsp).state
    # a valid tour starts and ends at the start city and visits every city once
    assert solution[0] == solution[-1] == 0
    assert set(solution) == set(cities)
    # the MST heuristic is admissible, so A* finds the optimal tour cost
    assert tsp.value(solution) == pytest.approx(3 + 5 ** 0.5)


def test_pour_problem():
    # the classic two-jug puzzle: with jugs of capacity 3 and 5, measure out 4
    problem = PourProblem(initial=(0, 0), goals={4}, capacities=(3, 5))
    solution = breadth_first_graph_search(problem)
    assert solution is not None
    assert any(level == 4 for level in solution.state)


def test_n_puzzle():
    # NPuzzle generalizes EightPuzzle; a one-move-from-goal 3x3 instance is solved by A*
    npuzzle = NPuzzle(initial=(1, 2, 3, 4, 5, 6, 0, 7, 8), size=3, shuffle=0)
    assert astar_search(npuzzle).solution() == ['RIGHT', 'RIGHT']
    # a shuffled instance is always solvable and reaches the goal
    solved = astar_search(NPuzzle(size=3, shuffle=15))
    assert solved is not None


def test_find_blank_square():
    assert eight_puzzle.find_blank_square((0, 1, 2, 3, 4, 5, 6, 7, 8)) == 0
    assert eight_puzzle.find_blank_square((6, 3, 5, 1, 8, 4, 2, 0, 7)) == 7
    assert eight_puzzle.find_blank_square((3, 4, 1, 7, 6, 0, 2, 8, 5)) == 5
    assert eight_puzzle.find_blank_square((1, 8, 4, 7, 2, 6, 3, 0, 5)) == 7
    assert eight_puzzle.find_blank_square((4, 8, 1, 6, 0, 2, 3, 5, 7)) == 4
    assert eight_puzzle.find_blank_square((1, 0, 6, 8, 7, 5, 4, 2, 3)) == 1
    assert eight_puzzle.find_blank_square((1, 2, 3, 4, 5, 6, 7, 8, 0)) == 8


def test_actions():
    assert eight_puzzle.actions((0, 1, 2, 3, 4, 5, 6, 7, 8)) == ['DOWN', 'RIGHT']
    assert eight_puzzle.actions((6, 3, 5, 1, 8, 4, 2, 0, 7)) == ['UP', 'LEFT', 'RIGHT']
    assert eight_puzzle.actions((3, 4, 1, 7, 6, 0, 2, 8, 5)) == ['UP', 'DOWN', 'LEFT']
    assert eight_puzzle.actions((1, 8, 4, 7, 2, 6, 3, 0, 5)) == ['UP', 'LEFT', 'RIGHT']
    assert eight_puzzle.actions((4, 8, 1, 6, 0, 2, 3, 5, 7)) == ['UP', 'DOWN', 'LEFT', 'RIGHT']
    assert eight_puzzle.actions((1, 0, 6, 8, 7, 5, 4, 2, 3)) == ['DOWN', 'LEFT', 'RIGHT']
    assert eight_puzzle.actions((1, 2, 3, 4, 5, 6, 7, 8, 0)) == ['UP', 'LEFT']


def test_result():
    assert eight_puzzle.result((0, 1, 2, 3, 4, 5, 6, 7, 8), 'DOWN') == (3, 1, 2, 0, 4, 5, 6, 7, 8)
    assert eight_puzzle.result((6, 3, 5, 1, 8, 4, 2, 0, 7), 'LEFT') == (6, 3, 5, 1, 8, 4, 0, 2, 7)
    assert eight_puzzle.result((3, 4, 1, 7, 6, 0, 2, 8, 5), 'UP') == (3, 4, 0, 7, 6, 1, 2, 8, 5)
    assert eight_puzzle.result((1, 8, 4, 7, 2, 6, 3, 0, 5), 'RIGHT') == (1, 8, 4, 7, 2, 6, 3, 5, 0)
    assert eight_puzzle.result((4, 8, 1, 6, 0, 2, 3, 5, 7), 'LEFT') == (4, 8, 1, 0, 6, 2, 3, 5, 7)
    assert eight_puzzle.result((1, 0, 6, 8, 7, 5, 4, 2, 3), 'DOWN') == (1, 7, 6, 8, 0, 5, 4, 2, 3)
    assert eight_puzzle.result((1, 2, 3, 4, 5, 6, 7, 8, 0), 'UP') == (1, 2, 3, 4, 5, 0, 7, 8, 6)
    assert eight_puzzle.result((4, 8, 1, 6, 0, 2, 3, 5, 7), 'RIGHT') == (4, 8, 1, 6, 2, 0, 3, 5, 7)


def test_goal_test():
    assert not eight_puzzle.goal_test((0, 1, 2, 3, 4, 5, 6, 7, 8))
    assert not eight_puzzle.goal_test((6, 3, 5, 1, 8, 4, 2, 0, 7))
    assert not eight_puzzle.goal_test((3, 4, 1, 7, 6, 0, 2, 8, 5))
    assert eight_puzzle.goal_test((1, 2, 3, 4, 5, 6, 7, 8, 0))
    assert not eight_puzzle2.goal_test((4, 8, 1, 6, 0, 2, 3, 5, 7))
    assert not eight_puzzle2.goal_test((3, 4, 1, 7, 6, 0, 2, 8, 5))
    assert not eight_puzzle2.goal_test((1, 2, 3, 4, 5, 6, 7, 8, 0))
    assert eight_puzzle2.goal_test((0, 1, 2, 3, 4, 5, 6, 7, 8))
    assert n_queens.goal_test((7, 3, 0, 2, 5, 1, 6, 4))
    assert n_queens.goal_test((0, 4, 7, 5, 2, 6, 1, 3))
    assert n_queens.goal_test((7, 1, 3, 0, 6, 4, 2, 5))
    assert not n_queens.goal_test((0, 1, 2, 3, 4, 5, 6, 7))


def test_check_solvability():
    assert eight_puzzle.check_solvability((0, 1, 2, 3, 4, 5, 6, 7, 8))
    assert eight_puzzle.check_solvability((6, 3, 5, 1, 8, 4, 2, 0, 7))
    assert eight_puzzle.check_solvability((3, 4, 1, 7, 6, 0, 2, 8, 5))
    assert eight_puzzle.check_solvability((1, 8, 4, 7, 2, 6, 3, 0, 5))
    assert eight_puzzle.check_solvability((4, 8, 1, 6, 0, 2, 3, 5, 7))
    assert eight_puzzle.check_solvability((1, 0, 6, 8, 7, 5, 4, 2, 3))
    assert eight_puzzle.check_solvability((1, 2, 3, 4, 5, 6, 7, 8, 0))
    assert not eight_puzzle.check_solvability((1, 2, 3, 4, 5, 6, 8, 7, 0))
    assert not eight_puzzle.check_solvability((1, 0, 3, 2, 4, 5, 6, 7, 8))
    assert not eight_puzzle.check_solvability((7, 0, 2, 8, 5, 3, 6, 4, 1))


def test_conflict():
    assert not n_queens.conflict(7, 0, 1, 1)
    assert not n_queens.conflict(0, 3, 6, 4)
    assert not n_queens.conflict(2, 6, 5, 7)
    assert not n_queens.conflict(2, 4, 1, 6)
    assert n_queens.conflict(0, 0, 1, 1)
    assert n_queens.conflict(4, 3, 4, 4)
    assert n_queens.conflict(6, 5, 5, 6)
    assert n_queens.conflict(0, 6, 1, 7)


def test_recursive_best_first_search():
    assert recursive_best_first_search(
        romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']
    assert recursive_best_first_search(
        EightPuzzle((2, 4, 3, 1, 5, 6, 7, 8, 0))).solution() == [
               'UP', 'LEFT', 'UP', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT', 'DOWN']

    def manhattan(node):
        state = node.state
        index_goal = {0: [2, 2], 1: [0, 0], 2: [0, 1], 3: [0, 2], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [2, 0], 8: [2, 1]}
        index_state = {}
        index = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

        for i in range(len(state)):
            index_state[state[i]] = index[i]

        mhd = 0

        for i in range(8):
            for j in range(2):
                mhd = abs(index_goal[i][j] - index_state[i][j]) + mhd

        return mhd

    assert recursive_best_first_search(
        EightPuzzle((2, 4, 3, 1, 5, 6, 7, 8, 0)), h=manhattan).solution() == [
               'LEFT', 'UP', 'UP', 'LEFT', 'DOWN', 'RIGHT', 'DOWN', 'UP', 'DOWN', 'RIGHT']


def test_hill_climbing():
    prob = PeakFindingProblem((0, 0), [[0, 5, 10, 20],
                                       [-3, 7, 11, 5]])
    assert hill_climbing(prob) == (0, 3)
    prob = PeakFindingProblem((0, 0), [[0, 5, 10, 8],
                                       [-3, 7, 9, 999],
                                       [1, 2, 5, 11]])
    assert hill_climbing(prob) == (0, 2)
    prob = PeakFindingProblem((2, 0), [[0, 5, 10, 8],
                                       [-3, 7, 9, 999],
                                       [1, 2, 5, 11]])
    assert hill_climbing(prob) == (1, 3)


def test_simulated_annealing():
    prob = PeakFindingProblem((0, 0), [[0, 5, 10, 20],
                                       [-3, 7, 11, 5]], directions4)
    sols = {prob.value(simulated_annealing(prob)) for _ in range(100)}
    assert max(sols) == 20
    prob = PeakFindingProblem((0, 0), [[0, 5, 10, 8],
                                       [-3, 7, 9, 999],
                                       [1, 2, 5, 11]], directions8)
    sols = {prob.value(simulated_annealing(prob)) for i in range(100)}
    assert max(sols) == 999


def test_BoggleFinder():
    board = list('SARTELNID')
    """
    >>> print_boggle(board)
        S  A  R
        T  E  L
        N  I  D
    """
    f = BoggleFinder(board)
    assert len(f) == 206


def test_and_or_graph_search():
    def run_plan(state, problem, plan):
        if problem.goal_test(state):
            return True
        if len(plan) != 2:
            return False
        predicate = lambda x: run_plan(x, problem, plan[1][x])
        return all(predicate(r) for r in problem.result(state, plan[0]))

    plan = and_or_graph_search(vacuum_world)
    assert run_plan('State_1', vacuum_world, plan)


def test_online_dfs_agent():
    odfs_agent = OnlineDFSAgent(LRTA_problem)
    # each call returns a single legal action (or None at the goal)
    first = odfs_agent('State_3')
    assert first in ['Right', 'Left']
    assert odfs_agent('State_5') is None

    # driving the agent through the environment must reach the goal and stop,
    # only ever issuing actions that are legal in the current state
    odfs_agent = OnlineDFSAgent(LRTA_problem)
    graph_dict = one_dim_state_space.graph_dict
    state = LRTA_problem.initial
    action = odfs_agent(state)
    for _ in range(40):
        if action is None:
            break
        assert action in graph_dict[state]
        state = graph_dict[state][action]
        action = odfs_agent(state)
    assert state == LRTA_problem.goal
    assert action is None


def test_LRTAStarAgent():
    lrta_agent = LRTAStarAgent(LRTA_problem)
    assert lrta_agent('State_3') == 'Right'
    assert lrta_agent('State_4') == 'Left'
    assert lrta_agent('State_3') == 'Right'
    assert lrta_agent('State_4') == 'Right'
    assert lrta_agent('State_5') is None

    lrta_agent = LRTAStarAgent(LRTA_problem)
    assert lrta_agent('State_4') == 'Left'

    lrta_agent = LRTAStarAgent(LRTA_problem)
    assert lrta_agent('State_5') is None


def test_genetic_algorithm():
    # Graph coloring
    edges = {'A': [0, 1],
             'B': [0, 3],
             'C': [1, 2],
             'D': [2, 3]}

    def fitness(c):
        return sum(c[n1] != c[n2] for (n1, n2) in edges.values())

    solution_chars = GA_GraphColoringChars(edges, fitness)
    assert solution_chars == ['R', 'G', 'R', 'G'] or solution_chars == ['G', 'R', 'G', 'R']

    solution_bools = GA_GraphColoringBools(edges, fitness)
    assert solution_bools == [True, False, True, False] or solution_bools == [False, True, False, True]

    solution_ints = GA_GraphColoringInts(edges, fitness)
    assert solution_ints == [0, 1, 0, 1] or solution_ints == [1, 0, 1, 0]

    # Queens Problem
    gene_pool = range(8)
    population = init_population(100, gene_pool, 8)

    def fitness(q):
        non_attacking = 0
        for row1 in range(len(q)):
            for row2 in range(row1 + 1, len(q)):
                col1 = int(q[row1])
                col2 = int(q[row2])
                row_diff = row1 - row2
                col_diff = col1 - col2

                if col1 != col2 and row_diff != col_diff and row_diff != -col_diff:
                    non_attacking += 1

        return non_attacking

    solution = genetic_algorithm(population, fitness, gene_pool=gene_pool, f_thres=25)
    assert fitness(solution) >= 25


def GA_GraphColoringChars(edges, fitness):
    gene_pool = ['R', 'G']
    population = init_population(8, gene_pool, 4)

    return genetic_algorithm(population, fitness, gene_pool=gene_pool)


def GA_GraphColoringBools(edges, fitness):
    gene_pool = [True, False]
    population = init_population(8, gene_pool, 4)

    return genetic_algorithm(population, fitness, gene_pool=gene_pool)


def GA_GraphColoringInts(edges, fitness):
    population = init_population(8, [0, 1], 4)

    return genetic_algorithm(population, fitness)


def test_simple_problem_solving_agent():
    class vacuumAgent(SimpleProblemSolvingAgentProgram):
        def update_state(self, state, percept):
            return percept

        def formulate_goal(self, state):
            goal = [state7, state8]
            return goal

        def formulate_problem(self, state, goal):
            problem = state
            return problem

        def search(self, problem):
            if problem == state1:
                seq = ["Suck", "Right", "Suck"]
            elif problem == state2:
                seq = ["Suck", "Left", "Suck"]
            elif problem == state3:
                seq = ["Right", "Suck"]
            elif problem == state4:
                seq = ["Suck"]
            elif problem == state5:
                seq = ["Suck"]
            elif problem == state6:
                seq = ["Left", "Suck"]
            return seq

    state1 = [(0, 0), [(0, 0), "Dirty"], [(1, 0), ["Dirty"]]]
    state2 = [(1, 0), [(0, 0), "Dirty"], [(1, 0), ["Dirty"]]]
    state3 = [(0, 0), [(0, 0), "Clean"], [(1, 0), ["Dirty"]]]
    state4 = [(1, 0), [(0, 0), "Clean"], [(1, 0), ["Dirty"]]]
    state5 = [(0, 0), [(0, 0), "Dirty"], [(1, 0), ["Clean"]]]
    state6 = [(1, 0), [(0, 0), "Dirty"], [(1, 0), ["Clean"]]]
    state7 = [(0, 0), [(0, 0), "Clean"], [(1, 0), ["Clean"]]]
    state8 = [(1, 0), [(0, 0), "Clean"], [(1, 0), ["Clean"]]]

    a = vacuumAgent(state1)

    assert a(state6) == "Left"
    assert a(state1) == "Suck"
    assert a(state3) == "Right"


# TODO: for .ipynb:
"""
>>> compare_graph_searchers()
    Searcher                      romania_map(A, B)        romania_map(O, N)         australia_map
    breadth_first_tree_search     <  21/  22/  59/B>   <1158/1159/3288/N>    <   7/   8/  22/WA>
    breadth_first_graph_search          <   7/  11/  18/B>   <  19/  20/  45/N>    <   2/   6/   8/WA>
    depth_first_graph_search      <   8/   9/  20/B>   <  16/  17/  38/N>    <   4/   5/  11/WA>
    iterative_deepening_search    <  11/  33/  31/B>   < 656/1815/1812/N>    <   3/  11/  11/WA>
    depth_limited_search          <  54/  65/ 185/B>   < 387/1012/1125/N>    <  50/  54/ 200/WA>
    recursive_best_first_search   <   5/   6/  15/B>   <5887/5888/16532/N>   <  11/12/  43/WA>

>>> ' '.join(f.words())
'LID LARES DEAL LIE DIETS LIN LINT TIL TIN RATED ERAS LATEN DEAR TIE LINE INTER
STEAL LATED LAST TAR SAL DITES RALES SAE RETS TAE RAT RAS SAT IDLE TILDES LEAST
IDEAS LITE SATED TINED LEST LIT RASE RENTS TINEA EDIT EDITS NITES ALES LATE
LETS RELIT TINES LEI LAT ELINT LATI SENT TARED DINE STAR SEAR NEST LITAS TIED
SEAT SERAL RATE DINT DEL DEN SEAL TIER TIES NET SALINE DILATE EAST TIDES LINTER
NEAR LITS ELINTS DENI RASED SERA TILE NEAT DERAT IDLEST NIDE LIEN STARED LIER
LIES SETA NITS TINE DITAS ALINE SATIN TAS ASTER LEAS TSAR LAR NITE RALE LAS
REAL NITER ATE RES RATEL IDEA RET IDEAL REI RATS STALE DENT RED IDES ALIEN SET
TEL SER TEN TEA TED SALE TALE STILE ARES SEA TILDE SEN SEL ALINES SEI LASE
DINES ILEA LINES ELD TIDE RENT DIEL STELA TAEL STALED EARL LEA TILES TILER LED
ETA TALI ALE LASED TELA LET IDLER REIN ALIT ITS NIDES DIN DIE DENTS STIED LINER
LASTED RATINE ERA IDLES DIT RENTAL DINER SENTI TINEAL DEIL TEAR LITER LINTS
TEAL DIES EAR EAT ARLES SATE STARE DITS DELI DENTAL REST DITE DENTIL DINTS DITA
DIET LENT NETS NIL NIT SETAL LATS TARE ARE SATI'

>>> boggle_hill_climbing(list('ABCDEFGHI'), verbose=False)
(['E', 'P', 'R', 'D', 'O', 'A', 'G', 'S', 'T'], 123)
"""

def test_plan_route():
    dim = 4
    allowed = [[i, j] for i in range(1, dim + 1) for j in range(1, dim + 1)]
    start = WumpusPosition(1, 1, 'UP')

    # a route to a goal cell: the planned actions must actually reach it
    problem = PlanRoute(start, [[3, 3]], allowed, dim)
    state = start
    for action in astar_search(problem).solution():
        state = problem.result(state, action)
    assert list(state.get_location()) == [3, 3]
    # the A* search must not mutate the initial state
    assert start.get_location() == (1, 1) and start.get_orientation() == 'UP'

    # a position goal also constrains the final orientation
    problem = PlanRoute(start, [WumpusPosition(2, 1, 'RIGHT')], allowed, dim)
    state = start
    for action in astar_search(problem).solution():
        state = problem.result(state, action)
    assert state.get_location() == (2, 1) and state.get_orientation() == 'RIGHT'

    # forward into a cell that is not allowed (unsafe) leaves the agent in place
    problem = PlanRoute(start, [[2, 2]], [[1, 1]], dim)
    assert problem.result(WumpusPosition(1, 1, 'RIGHT'), 'Forward').get_location() == (1, 1)


def test_grid_problem():
    # 5x5 grid with a wall blocking the direct route (gap at y=4)
    walls = [(2, y) for y in range(4)]
    problem = GridProblem((0, 0), (4, 0), 5, 5, obstacles=walls)
    astar = astar_search(problem)
    bfs = breadth_first_graph_search(problem)
    assert astar is not None
    assert astar.path_cost == bfs.path_cost                            # both optimal
    assert all(problem.passable(node.state) for node in astar.path())  # avoids walls
    # a goal walled off on every side is unreachable
    boxed = GridProblem((0, 0), (4, 4), 5, 5, obstacles=[(3, 4), (4, 3)])
    assert astar_search(boxed) is None


def test_grid_search_visualization():
    import matplotlib
    matplotlib.use('Agg')
    from aima.notebook_utils import grid_search_steps, plot_grid_search
    walls = [(3, y) for y in range(7)]
    problem = GridProblem((0, 0), (6, 0), 10, 10, obstacles=walls)
    expl_bfs, path_bfs = grid_search_steps(problem, 'bfs')
    expl_astar, path_astar = grid_search_steps(problem, 'astar')
    assert path_bfs[0] == path_astar[0] == (0, 0)
    assert path_bfs[-1] == path_astar[-1] == (6, 0)
    assert len(path_astar) == len(path_bfs)            # same optimal path length
    assert len(expl_astar) <= len(expl_bfs)            # informed search is more focused
    assert plot_grid_search(problem, expl_astar, path_astar) is not None


def test_local_search_variants():
    # a unimodal grid (value = x + y + 1, single peak at (2, 2)): every local-search
    # variant climbs to it. These algorithms have no pseudocode in the book (#1151).
    random.seed(0)
    grid = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert stochastic_hill_climbing(PeakFindingProblem((0, 0), grid)) == (2, 2)
    assert first_choice_hill_climbing(PeakFindingProblem((0, 0), grid)) == (2, 2)
    assert local_beam_search(PeakFindingProblem((0, 0), grid), k=3) == (2, 2)
    n, m = len(grid), len(grid[0])
    best = random_restart_hill_climbing(
        PeakFindingProblem((0, 0), grid),
        lambda: (random.randrange(n), random.randrange(m)), restarts=5)
    assert best == (2, 2)


def test_node_path_states():
    # the full root->goal path is one call away from the returned Node (#1068)
    node = astar_search(GraphProblem('Arad', 'Bucharest', romania_map))
    states = node.path_states()
    assert states == [n.state for n in node.path()]
    assert states[0] == 'Arad' and states[-1] == 'Bucharest'
    assert len(node.solution()) == len(states) - 1   # actions are the path's edges


if __name__ == '__main__':
    pytest.main()
