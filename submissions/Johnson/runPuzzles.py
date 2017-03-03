import search
import submissions.Johnson.puzzles as pz

def compare_searchers(problems, header, searchers=[]):
    def do(searcher, problem):
        p = search.InstrumentedProblem(problem)
        goalNode = searcher(p)
        return p, goalNode.path_cost
    table = [[search.name(s)] + [do(s, p) for p in problems] for s in searchers]
    search.print_table(table, header)

    class GapSwitch(search.Problem):

        #A 4x4 grid, each square unmarked at the start.
        grid = (['U', 'U', 'U', 'U'],
                ['U', 'U', 'U', 'U'],
                ['U', 'U', 'U', 'U'],
                ['U', 'U', 'U', 'U'])

        #defines the state using row number first, then the spaces between marked squares required in that row,
        #  column number, and the spaces between marked squares required in that column
        state = ((2,2),(4,1))

        def actions(self, state):
            return ['M', 'U']

        def result(self, state, action):
            if action == 'up':
                return 'on'
            else:
                return 'off'

        def start(self, state, grid):
            if

        def goal_test(self, state):
            return state == 'on'

        def h(self, node):
            state = node.state
            if self.goal_test(state):
                return 0
            else:
                return 1

    switch_puzzle = LightSwitch('off')
    switch_puzzle.label = 'Light Switch'

compare_searchers(
    problems=pz.myPuzzles,
    header=['Searcher',
        '(<succ/goal/stat/fina>, cost)'
    ],
    searchers=[
        search.breadth_first_search,
        search.depth_first_graph_search,
        search.uniform_cost_search,
        search.astar_search,
        search.recursive_best_first_search

    ]
)