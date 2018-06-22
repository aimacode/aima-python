from mdp import *

sequential_decision_environment_1 = GridMDP([[-0.1, -0.1, -0.1, +1],
                                             [-0.1, None, -0.1, -1],
                                             [-0.1, -0.1, -0.1, -0.1]],
                                            terminals=[(3, 2), (3, 1)])

sequential_decision_environment_2 = GridMDP([[-2, -2, -2, +1],
                                             [-2, None, -2, -1],
                                             [-2, -2, -2, -2]],
                                            terminals=[(3, 2), (3, 1)])

sequential_decision_environment_3 = GridMDP([[-1.0, -0.1, -0.1, -0.1, -0.1, 0.5], 
                                             [-0.1, None, None, -0.5, -0.1, -0.1], 
                                             [-0.1, None, 1.0, 3.0, None, -0.1], 
                                             [-0.1, -0.1, -0.1, None, None, -0.1], 
                                             [0.5, -0.1, -0.1, -0.1, -0.1, -1.0]],
                                            terminals=[(2, 2), (3, 2), (0, 4), (5, 0)])

def test_value_iteration():
    assert value_iteration(sequential_decision_environment, .01) == {
        (3, 2): 1.0, (3, 1): -1.0,
        (3, 0): 0.12958868267972745, (0, 1): 0.39810203830605462,
        (0, 2): 0.50928545646220924, (1, 0): 0.25348746162470537,
        (0, 0): 0.29543540628363629, (1, 2): 0.64958064617168676,
        (2, 0): 0.34461306281476806, (2, 1): 0.48643676237737926,
        (2, 2): 0.79536093684710951}

    assert value_iteration(sequential_decision_environment_1, .01) == {
        (3, 2): 1.0, (3, 1): -1.0,  
        (3, 0): -0.0897388258468311, (0, 1): 0.146419707398967840, 
        (0, 2): 0.30596200514385086, (1, 0): 0.010092796415625799,
        (0, 0): 0.00633408092008296, (1, 2): 0.507390193380827400, 
        (2, 0): 0.15072242145212010, (2, 1): 0.358309043654212570, 
        (2, 2): 0.71675493618997840}

    assert value_iteration(sequential_decision_environment_2, .01) == {
        (3, 2): 1.0, (3, 1): -1.0, 
        (3, 0): -3.5141584808407855, (0, 1): -7.8000009574737180,
        (0, 2): -6.1064293596058830, (1, 0): -7.1012549580376760,
        (0, 0): -8.5872244532783200, (1, 2): -3.9653547121245810,
        (2, 0): -5.3099468802901630, (2, 1): -3.3543366255753995,
        (2, 2): -1.7383376462930498}

    assert value_iteration(sequential_decision_environment_3, .01) == {
        (0, 0): 4.350592130345558, (0, 1): 3.640700980321895, (0, 2): 3.0734806370346943, (0, 3): 2.5754335063434937, (0, 4): -1.0,
        (1, 0): 3.640700980321895, (1, 1): 3.129579352304856, (1, 4): 2.0787517066719916,
        (2, 0): 3.0259220379893352, (2, 1): 2.5926103577982897, (2, 2): 1.0, (2, 4): 2.507774181360808,
        (3, 0): 2.5336747364500076, (3, 2): 3.0, (3, 3): 2.292172805400873, (3, 4): 2.996383110867515,
        (4, 0): 2.1014575936349886, (4, 3): 3.1297590518608907, (4, 4): 3.6408806798779287,
        (5, 0): -1.0, (5, 1): 2.5756132058995282, (5, 2): 3.0736603365907276, (5, 3): 3.6408806798779287, (5, 4): 4.350771829901593}


def test_policy_iteration():
    assert policy_iteration(sequential_decision_environment) == {
        (0, 0): (0, 1), (0, 1): (0, 1), (0, 2): (1, 0),
        (1, 0): (1, 0), (1, 2): (1, 0), (2, 0): (0, 1),
        (2, 1): (0, 1), (2, 2): (1, 0), (3, 0): (-1, 0),
        (3, 1): None, (3, 2): None}

    assert policy_iteration(sequential_decision_environment_1) == {
        (0, 0): (0, 1), (0, 1): (0, 1), (0, 2): (1, 0),
        (1, 0): (1, 0), (1, 2): (1, 0), (2, 0): (0, 1),
        (2, 1): (0, 1), (2, 2): (1, 0), (3, 0): (-1, 0),
        (3, 1): None, (3, 2): None}

    assert policy_iteration(sequential_decision_environment_2) == {
        (0, 0): (1, 0), (0, 1): (0, 1), (0, 2): (1, 0),
        (1, 0): (1, 0), (1, 2): (1, 0), (2, 0): (1, 0),
        (2, 1): (1, 0), (2, 2): (1, 0), (3, 0): (0, 1),
        (3, 1): None, (3, 2): None}


def test_best_policy():
    pi = best_policy(sequential_decision_environment,
                     value_iteration(sequential_decision_environment, .01))
    assert sequential_decision_environment.to_arrows(pi) == [['>', '>', '>', '.'],
                                                             ['^', None, '^', '.'],
                                                             ['^', '>', '^', '<']]

    pi_1 = best_policy(sequential_decision_environment_1,
                     value_iteration(sequential_decision_environment_1, .01))
    assert sequential_decision_environment_1.to_arrows(pi_1) == [['>', '>', '>', '.'],
                                                                 ['^', None, '^', '.'],
                                                                 ['^', '>', '^', '<']]

    pi_2 = best_policy(sequential_decision_environment_2,
                     value_iteration(sequential_decision_environment_2, .01))
    assert sequential_decision_environment_2.to_arrows(pi_2) == [['>', '>', '>', '.'],
                                                                 ['^', None, '>', '.'],
                                                                 ['>', '>', '>', '^']]

    pi_3 = best_policy(sequential_decision_environment_3,
                     value_iteration(sequential_decision_environment_3, .01))
    assert sequential_decision_environment_3.to_arrows(pi_3) == [['.', '>', '>', '>', '>', '>'], 
                                                                 ['v', None, None, '>', '>', '^'], 
                                                                 ['v', None, '.', '.', None, '^'], 
                                                                 ['v', '<', 'v', None, None, '^'], 
                                                                 ['<', '<', '<', '<', '<', '.']]                                                               


def test_transition_model():
    transition_model = { 'a' : {   'plan1' : [(0.2, 'a'), (0.3, 'b'), (0.3, 'c'), (0.2, 'd')],
                    'plan2' : [(0.4, 'a'), (0.15, 'b'), (0.45, 'c')],
                    'plan3' : [(0.2, 'a'), (0.5, 'b'), (0.3, 'c')],
                },
          'b' : {   'plan1' : [(0.2, 'a'), (0.6, 'b'), (0.2, 'c'), (0.1, 'd')],
                    'plan2' : [(0.6, 'a'), (0.2, 'b'), (0.1, 'c'), (0.1, 'd')],
                    'plan3' : [(0.3, 'a'), (0.3, 'b'), (0.4, 'c')],
                },
          'c' : {   'plan1' : [(0.3, 'a'), (0.5, 'b'), (0.1, 'c'), (0.1, 'd')],
                    'plan2' : [(0.5, 'a'), (0.3, 'b'), (0.1, 'c'), (0.1, 'd')],
                    'plan3' : [(0.1, 'a'), (0.3, 'b'), (0.1, 'c'), (0.5, 'd')],
                },
        }

    mdp = MDP(init="a", actlist={"plan1","plan2", "plan3"}, terminals={"d"}, states={"a","b","c", "d"}, transitions=transition_model)

    assert mdp.T("a","plan3") == [(0.2, 'a'), (0.5, 'b'), (0.3, 'c')]
    assert mdp.T("b","plan2") == [(0.6, 'a'), (0.2, 'b'), (0.1, 'c'), (0.1, 'd')]
    assert mdp.T("c","plan1") == [(0.3, 'a'), (0.5, 'b'), (0.1, 'c'), (0.1, 'd')]


def test_pomdp_value_iteration():
    t_prob = [[[0.65, 0.35], [0.65, 0.35]], [[0.65, 0.35], [0.65, 0.35]], [[1.0, 0.0], [0.0, 1.0]]]
    e_prob = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.8, 0.2], [0.3, 0.7]]]
    rewards = [[5, -10], [-20, 5], [-1, -1]]

    gamma = 0.95
    actions = ('0', '1', '2')
    states = ('0', '1')

    pomdp = POMDP(actions, t_prob, e_prob, rewards, states, gamma)
    utility = pomdp_value_iteration(pomdp, epsilon=5)
    
    for _, v in utility.items():
        sum_ = 0
        for element in v:
            sum_ += sum(element)
    # exact value was found to be -9.73231
    assert -9.76 < sum_ < -9.70


def test_pomdp_value_iteration2():
    t_prob = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[1.0, 0.0], [0.0, 1.0]]]
    e_prob = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.85, 0.15], [0.15, 0.85]]]
    rewards = [[-100, 10], [10, -100], [-1, -1]]

    gamma = 0.95
    actions = ('0', '1', '2')
    states = ('0', '1')

    pomdp = POMDP(actions, t_prob, e_prob, rewards, states, gamma)
    utility = pomdp_value_iteration(pomdp, epsilon=100)

    for _, v in utility.items():
        sum_ = 0
        for element in v:
            sum_ += sum(element)
    # exact value was found to be -77.28259
    assert -77.31 < sum_ < -77.25
