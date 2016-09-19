import search
import submissions.McLean.puzzles as pz

def compare_searchers(problems, header, searchers=[]):
    def do(searcher, problem):
        p = search.InstrumentedProblem(problem)
        goalNode = searcher(p)
        return p, goalNode.path_cost
    table = [[search.name(s)] + [do(s, p) for p in problems] for s in searchers]
    search.print_table(table, header)

compare_searchers(
    problems=pz.myPuzzles,
    header=['Searcher',
        '(<succ/goal/stat/fina>, cost)'
    ],
    searchers=[
        search.breadth_first_search,
        search.depth_first_graph_search,
    ]
)