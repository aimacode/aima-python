import pytest
from search import *

class Graph(Problem):

    """
    Graph class to test uninformed search algorithms which work on graphs with path costs.
    """

    def __init__(self, initial, goal=None, paths={}, bidirectional = False):
        """
        The constructor takes as input the initial state, list of goal states and a dictionary representing a list of tuples which contiains the action and path cost
        """
        #Make a dictionary of actions
        action_dict = {}
        for state in paths.keys():
            action_dict[state] = {}
            for next_state, path_cost in paths[state]:
                action_dict[state][next_state] = path_cost
            if bidirectional:
                if next_state not in action_dict.keys():
                    action_dict[next_state]={}
                action_dict[next_state][state] = path_cost

        update(self, initial=initial, goal=goal, action_dict=action_dict)

    def actions(self, state):
        """
        returns the possible actions to take as a list of strings representing the state that action leads to
        """
        return [ action for action in self.action_dict[state] ]

    def result(self, state, action):
        """
        Return the state that results from executing the given action
        """
        #Make sure the action is in actions(state)
        assert is_in(action, self.actions(state))
        return action

    def goal_test(self, state):
        """
        Return True if the state is a goal.
        """
        return is_in(state, self.goal)

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + self.action_dict[state1][state2]

Fig[3, 12] = Graph('A', ['G'], {'A':[('B', 1), ('C', 1)],
                                'B':[('D', 1), ('E', 1)],
                                'C':[('F', 1), ('G', 1)],
                                'D':[],
                                'E':[],
                                'F':[],
                                'G':[]})

def test_breadth_first_tree_search():
    solution_node = breadth_first_tree_search(Fig[3, 12])
    assert solution_node.solution() == ['C', 'G']
    assert [node.action for node in solution_node.path()] == [None, 'C', 'G']
    #Test BFS if no goal is present
    Fig[3, 12].goal = []
    assert breadth_first_tree_search(Fig[3, 12]) is None
    Fig[3, 12].goal = ['G']

if __name__ == '__main__':
    pytest.main()
