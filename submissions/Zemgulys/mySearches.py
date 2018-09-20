import search

dupage_map = search.UndirectedGraph(dict(
    Elmhurst=dict(VillaPark=10, OakBrook=17, Addison=11),
    VillaPark=dict(Elmhurst=10, OakBrook=13, Addison=7, Lombard=4),
    OakBrook=dict(VillaPark=13, Elmhurst=17, Westmont=11, Hinsdale=12),
    Addison=dict(Elmhurst=11, VillaPark=7, GH=12, WD=5),
    Lombard=dict(VillaPark=4, GH=13, GE=10),
    Westmont=dict(OakBrook=11, Hinsdale=7, DG=5),
    GH=dict(Addison=12, Lombard=13, CS=14),
    Hinsdale=dict(Westmont=7, OakBrook=12),
    GE=dict(Lombard=10),
    Lisle=dict(OT=27, Warrenville=12, DG=6),
    Warrenville=dict(Lisle=12, Winfield=13),
    Winfield=dict(Warrenville=13, Wheaton=9),
    Wheaton=dict(OT=26, Winfield=9, BD=34, WC=14),
    DG=dict(Westmont=5, Lisle=6),
    OT=dict(Wheaton=26, Lisle=27),
    CS=dict(GH=14),
    WC=dict(Wheaton=14, BD=10),
    WD=dict(Addison=5),
    BD=dict(WC=10, Wheaton=34)
))

# dupage_puzzle = search.GraphProblem('Addison', 'OakBrook', dupage_map)
# dupage_puzzle = search.GraphProblem('GE', 'Wheaton', dupage_map)
dupage_puzzle = search.GraphProblem('WD', 'WC', dupage_map)

dupage_puzzle.label = 'Dupage'


# A trivial Problem definition
class LightSwitch(search.Problem):
    def actions(self, state):
        return ['up', 'down']

    def result(self, state, action):
        if action == 'up':
            return 'on'
        else:
            return 'off'

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

mySearches = [
    dupage_puzzle,
    # switch_puzzle,
]

mySearchMethods = []
