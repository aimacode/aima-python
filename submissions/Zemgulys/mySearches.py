import search

maze = (dict(
    A=dict(B=0, C=0),
    B=dict(A=0, D=0, F=0),
    C=dict(A=0, E=0, G=0),
    D=dict(B=0, E=0),
    E=dict(D=0, C=0, H=0),
    F=dict(B=0, G=0, H=0),
    G=dict(C=0, F=0),
    H=dict(E=0, F=0)
))


class hamiltonMaze(search.Problem):

    def __init__(self, initial, goal, map):
        self.initial = initial
        self.goal = goal
        self.map = map

        self.expandedNodes = []
        self.invalidMoves = []
        self.validMoves = []

    def actions(self, state):
        curr = self.map[state]

        if len(self.expandedNodes) < len(maze):
            self.expandedNodes.append(curr)
        keys = curr.keys()
        if keys in self.expandedNodes:
            self.invalidMoves.append(keys)
        else:
            self.validMoves.append(keys)
        return keys

    def result(self, state, action):
        if action in self.expandedNodes:
            self.invalidMoves.append(action)
        else:
            return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        curr = self.map[state1]
        cost = curr[state2]
        return c + cost

    def h(self, node):
        state = node.state
        if self.goal_test(state):
            return 0
        else:
            return 1


hamilton_maze = hamiltonMaze('A', 'H', maze)
hamilton_maze.label = 'Hamilton Maze'
mySearches = [

    hamilton_maze
]

mySearchMethods = []
