import pytest
from search import *  # noqa


romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
vacumm_world = GraphProblemStochastic('State_1', ['State_7', 'State_8'], vacumm_world)
LRTA_problem = OnlineSearchProblem('State_3', 'State_5', one_dim_state_space)


def test_breadth_first_tree_search():
    assert breadth_first_tree_search(romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']


def test_breadth_first_search():
    assert breadth_first_search(romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']


def test_uniform_cost_search():
    assert uniform_cost_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']


def test_depth_first_graph_search():
    solution = depth_first_graph_search(romania_problem).solution()
    assert solution[-1] == 'Bucharest'


def test_iterative_deepening_search():
    assert iterative_deepening_search(romania_problem).solution() == ['Sibiu', 'Fagaras', 'Bucharest']


def test_depth_limited_search():
    solution_3 = depth_limited_search(romania_problem, 3).solution()
    assert solution_3[-1] == 'Bucharest'
    assert depth_limited_search(romania_problem, 2) == 'cutoff'
    solution_50 = depth_limited_search(romania_problem).solution()
    assert solution_50[-1] == 'Bucharest'


def test_astar_search():
    assert astar_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']


def test_recursive_best_first_search():
    assert recursive_best_first_search(romania_problem).solution() == ['Sibiu', 'Rimnicu', 'Pitesti', 'Bucharest']


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
        if len(plan) is not 2:
            return False
        predicate = lambda x : run_plan(x, problem, plan[1][x])
        return all(predicate(r) for r in problem.result(state, plan[0]))
    plan = and_or_graph_search(vacumm_world)
    assert run_plan('State_1', vacumm_world, plan)


def test_LRTAStarAgent():
    my_agent = LRTAStarAgent(LRTA_problem)
    assert my_agent('State_3') == 'Right'
    assert my_agent('State_4') == 'Left'
    assert my_agent('State_3') == 'Right'
    assert my_agent('State_4') == 'Right'
    assert my_agent('State_5') is None

    my_agent = LRTAStarAgent(LRTA_problem)
    assert my_agent('State_4') == 'Left'

    my_agent = LRTAStarAgent(LRTA_problem)
    assert my_agent('State_5') is None

# TODO: for .ipynb:
"""
>>> compare_graph_searchers()
    Searcher                      romania_map(A, B)        romania_map(O, N)         australia_map
    breadth_first_tree_search     <  21/  22/  59/B>   <1158/1159/3288/N>    <   7/   8/  22/WA>
    breadth_first_search          <   7/  11/  18/B>   <  19/  20/  45/N>    <   2/   6/   8/WA>
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

if __name__ == '__main__':
    pytest.main()
