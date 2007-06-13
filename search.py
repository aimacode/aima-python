"""Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""

from __future__ import generators
from utils import *
import agents
import math, random, sys, time, bisect, string

#______________________________________________________________________________

class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial; self.goal = goal
        
    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework."""
        abstract
    
    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state == self.goal
    
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        abstract
#______________________________________________________________________________
    
class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action, 
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1
            
    def __repr__(self):
        return "<Node %s>" % (self.state,)
    
    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        return [Node(next, self, act,
                     problem.path_cost(self.path_cost, self.state, act, next))
                for (act, next) in problem.successor(self.state)]

#______________________________________________________________________________

class SimpleProblemSolvingAgent(agents.Agent):
    """Abstract framework for problem-solving agent. [Fig. 3.1]"""
    def __init__(self):
        Agent.__init__(self)
        state = []
        seq = []

        def program(percept):
            state = self.update_state(state, percept)
            if not seq:
                goal = self.formulate_goal(state)
                problem = self.formulate_problem(state, goal)
                seq = self.search(problem)
            action = seq[0]
            seq[0:1] = []
            return action

        self.program = program

#______________________________________________________________________________
## Uninformed Search algorithms

def tree_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    Don't worry about repeated paths to a state. [Fig. 3.8]"""
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None

def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return tree_search(problem, FIFOQueue())
    
def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return tree_search(problem, Stack())

def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]"""
    closed = {}
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state): 
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))    
    return None

def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return graph_search(problem, FIFOQueue())
    
def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return graph_search(problem, Stack())

def depth_limited_search(problem, limit=50):
    "[Fig. 3.12]"
    def recursive_dls(node, problem, limit):
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result != None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        else:
            return None
    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    "[Fig. 3.13]"
    for depth in xrange(sys.maxint):
        result = depth_limited_search(problem, depth)
        if result is not 'cutoff':
            return result

#______________________________________________________________________________
# Informed (Heuristic) Search

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have depth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    return graph_search(problem, PriorityQueue(min, f))

greedy_best_first_graph_search = best_first_graph_search
    # Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search.
    Uses the pathmax trick: f(n) = max(f(n), g(n)+h(n))."""
    h = h or problem.h
    def f(n):
        return max(getattr(n, 'f', -infinity), n.path_cost + h(n))
    return best_first_graph_search(problem, f)

#______________________________________________________________________________
## Other search algorithms

def recursive_best_first_search(problem):
    "[Fig. 4.5]"
    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state): 
            return node
        successors = expand(node, problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + s.h, node.f)
        while True:
            successors.sort(lambda x,y: x.f - y.f) # Order by lowest f value
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            alternative = successors[1]
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result
    return RBFS(Node(problem.initial), infinity)


def hill_climbing(problem):
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better. [Fig. 4.11]"""
    current = Node(problem.initial)
    while True:
        neighbor = argmax(expand(node, problem), Node.value)
        if neighbor.value() <= current.value():
            return current.state
        current = neighbor

def exp_schedule(k=20, lam=0.005, limit=100):
    "One possible schedule function for simulated annealing"
    return lambda t: if_(t < limit, k * math.exp(-lam * t), 0)

def simulated_annealing(problem, schedule=exp_schedule()):
    "[Fig. 4.5]"
    current = Node(problem.initial)
    for t in xrange(sys.maxint):
        T = schedule(t)
        if T == 0:
            return current
        next = random.choice(expand(node. problem))
        delta_e = next.path_cost - current.path_cost
        if delta_e > 0 or probability(math.exp(delta_e/T)):
            current = next

def online_dfs_agent(a):
    "[Fig. 4.12]"
    pass #### more

def lrta_star_agent(a):
    "[Fig. 4.12]"
    pass #### more

#______________________________________________________________________________
# Genetic Algorithm

def genetic_search(problem, fitness_fn, ngen=1000, pmut=0.0, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires that the problem has a successor function that generates
    reasonable states, and that it has a path_cost function that scores states.
    We use the negative of the path_cost function, because costs are to be
    minimized, while genetic-algorithm expects a fitness_fn to be maximized."""
    states = [s for (a, s) in problem.successor(problem.initial_state)[:n]]
    random.shuffle(states)
    fitness_fn = lambda s: - problem.path_cost(0, s, None, s)
    return genetic_algorithm(states, fitness_fn, ngen, pmut)

def genetic_algorithm(population, fitness_fn, ngen=1000, pmut=0.0):
    """[Fig. 4.7]"""
    def reproduce(p1, p2):
        c = random.randrange(len(p1))
        return p1[:c] + p2[c:]

    for i in range(ngen):
        new_population = []
        for i in len(population):
            p1, p2 = random_weighted_selections(population, 2, fitness_fn)
            child = reproduce(p1, p2)
            if random.uniform(0,1) > pmut:
                child.mutate()
            new_population.append(child)
        population = new_population
    return argmax(population, fitness_fn)

def random_weighted_selection(seq, n, weight_fn):
    """Pick n elements of seq, weighted according to weight_fn.
    That is, apply weight_fn to each element of seq, add up the total.
    Then choose an element e with probability weight[e]/total.
    Repeat n times, with replacement. """
    totals = []; runningtotal = 0
    for item in seq:
        runningtotal += weight_fn(item)
        totals.append(runningtotal)
    selections = []
    for s in range(n):
        r = random.uniform(0, totals[-1])
        for i in range(len(seq)):
            if totals[i] > r:
                selections.append(seq[i])
                break
    return selections
    

#_____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.

#______________________________________________________________________________
# Graphs and Graph Problems

class Graph:
    """A graph connects nodes (verticies) by edges (links).  Each edge can also
    have a length associated with it.  The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})   
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C.  You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added.  You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B.  'Lengths' can actually be any object at 
    all, and nodes can be any hashable object."""

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed: self.make_undirected()

    def make_undirected(self):
        "Make a digraph into an undirected graph by adding symmetric edges."
        for a in self.dict.keys():
            for (b, distance) in self.dict[a].items():
                self.connect1(b, a, distance)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed: self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        "Add a link from A to B of given distance, in one direction only."
        self.dict.setdefault(A,{})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dict.setdefault(a, {})
        if b is None: return links
        else: return links.get(b)

    def nodes(self):
        "Return a list of nodes in the graph."
        return self.dict.keys()

def UndirectedGraph(dict=None):
    "Build a Graph where every edge (including future ones) goes both ways."
    return Graph(dict=dict, directed=False)

def RandomGraph(nodes=range(10), min_links=2, width=400, height=300,
                                curvature=lambda: random.uniform(1.1, 1.5)):
    """Construct a random graph, with the specified nodes, and random links.
    The nodes are laid out randomly on a (width x height) rectangle.
    Then each node is connected to the min_links nearest neighbors.
    Because inverse links are added, some nodes will have more connections.
    The distance between nodes is the hypotenuse times curvature(),
    where curvature() defaults to a random number between 1.1 and 1.5."""
    g = UndirectedGraph()
    g.locations = {}
    ## Build the cities
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    ## Build roads from each city to at least min_links nearest neighbors.
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]
                def distance_to_node(n):
                    if n is node or g.get(node,n): return infinity
                    return distance(g.locations[n], here)
                neighbor = argmin(nodes, distance_to_node)
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d)) 
    return g

romania = UndirectedGraph(Dict(
    A=Dict(Z=75, S=140, T=118),
    B=Dict(U=85, P=101, G=90, F=211),
    C=Dict(D=120, R=146, P=138),
    D=Dict(M=75),
    E=Dict(H=86),
    F=Dict(S=99),
    H=Dict(U=98),
    I=Dict(V=92, N=87),
    L=Dict(T=111, M=70),
    O=Dict(Z=71, S=151),
    P=Dict(R=97),
    R=Dict(S=80),
    U=Dict(V=142)))
romania.locations = Dict(
    A=( 91, 492),    B=(400, 327),    C=(253, 288),   D=(165, 299), 
    E=(562, 293),    F=(305, 449),    G=(375, 270),   H=(534, 350),
    I=(473, 506),    L=(165, 379),    M=(168, 339),   N=(406, 537), 
    O=(131, 571),    P=(320, 368),    R=(233, 410),   S=(207, 457), 
    T=( 94, 410),    U=(456, 350),    V=(509, 444),   Z=(108, 531))

australia = UndirectedGraph(Dict(
    T=Dict(),
    SA=Dict(WA=1, NT=1, Q=1, NSW=1, V=1),
    NT=Dict(WA=1, Q=1),
    NSW=Dict(Q=1, V=1)))
australia.locations = Dict(WA=(120, 24), NT=(135, 20), SA=(135, 30), 
                           Q=(145, 20), NSW=(145, 32), T=(145, 42), V=(145, 37))

class GraphProblem(Problem):
    "The problem of searching a graph from one node to another."
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def successor(self, A):
        "Return a list of (action, result) pairs."
        return [(B, B) for B in self.graph.get(A).keys()]

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity

#______________________________________________________________________________

#### NOTE: NQueensProblem not working properly yet.

class NQueensProblem(Problem):
    """The problem of placing N queens on an NxN board with none attacking
    each other.  A state is represented as an N-element array, where the
    a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of None means that the c-th column has not been
    filled in left.  We fill in columns left to right."""
    def __init__(self, N):
        self.N = N
        self.initial = [None] * N

    def successor(self, state): 
        "In the leftmost empty column, try all non-conflicting rows."
        if state[-1] is not None:
            return [] ## All columns filled; no successors
        else:
            def place(col, row):
                new = state[:]
                new[col] = row
                return new
            col = state.index(None)
            return [(row, place(col, row)) for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def conflicted(self, state, row, col):
        "Would placing a queen at (row, col) conflict with anything?"
        for c in range(col-1):
            if self.conflict(row, col, state[c], c):
                return True
        return False

    def conflict(self, row1, col1, row2, col2):
        "Would putting two queens in (row1, col1) and (row2, col2) conflict?"
        return (row1 == row2 ## same row
                or col1 == col2 ## same column
                or row1-col1 == row2-col2  ## same \ diagonal
                or row1+col1 == row2+col2) ## same / diagonal

    def goal_test(self, state):
        "Check if all columns filled, no conflicts."
        if state[-1] is None: 
            return False
        for c in range(len(state)):
            if self.conflicted(state, state[c], c):
                return False
        return True

#______________________________________________________________________________
## Inverse Boggle: Search for a high-scoring Boggle board. A good domain for
## iterative-repair and related search tehniques, as suggested by Justin Boyan.

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cubes16 = ['FORIXB', 'MOQABJ', 'GURILW', 'SETUPL',
           'CMPDAE', 'ACITAO', 'SLCRAE', 'ROMASH',
           'NODESW', 'HEFIYE', 'ONUDTK', 'TEVIGN',
           'ANEDVZ', 'PINESH', 'ABILYT', 'GKYLEU']

def random_boggle(n=4):
    """Return a random Boggle board of size n x n.
    We represent a board as a linear list of letters."""
    cubes = [cubes16[i % 16] for i in range(n*n)]
    random.shuffle(cubes)
    return map(random.choice, cubes)

## The best 5x5 board found by Boyan, with our word list this board scores
## 2274 words, for a score of 9837

boyan_best = list('RSTCSDEIAEGNLRPEATESMSSID')

def print_boggle(board):
    "Print the board in a 2-d array."
    n2 = len(board); n = exact_sqrt(n2)
    for i in range(n2):
        if i % n == 0: print
        if board[i] == 'Q': print 'Qu',
        else: print str(board[i]) + ' ',
    print
    
def boggle_neighbors(n2, cache={}):
    """"Return a list of lists, where the i-th element is the list of indexes
    for the neighbors of square i."""
    if cache.get(n2):
        return cache.get(n2)
    n = exact_sqrt(n2)
    neighbors = [None] * n2
    for i in range(n2):
        neighbors[i] = []
        on_top = i < n
        on_bottom = i >= n2 - n
        on_left = i % n == 0
        on_right = (i+1) % n == 0
        if not on_top:
            neighbors[i].append(i - n)
            if not on_left:  neighbors[i].append(i - n - 1)
            if not on_right: neighbors[i].append(i - n + 1)
        if not on_bottom:
            neighbors[i].append(i + n) 
            if not on_left:  neighbors[i].append(i + n - 1)
            if not on_right: neighbors[i].append(i + n + 1)
        if not on_left: neighbors[i].append(i - 1) 
        if not on_right: neighbors[i].append(i + 1)
    cache[n2] = neighbors
    return neighbors

def exact_sqrt(n2):
    "If n2 is a perfect square, return its square root, else raise error."
    n = int(math.sqrt(n2))
    assert n * n == n2
    return n

##_____________________________________________________________________________

class Wordlist:
    """This class holds a list of words. You can use (word in wordlist)
    to check if a word is in the list, or wordlist.lookup(prefix)
    to see if prefix starts any of the words in the list."""
    def __init__(self, filename, min_len=3):
        lines = open(filename).read().upper().split()
        self.words = [word for word in lines if len(word) >= min_len]
        self.words.sort()
        self.bounds = {}
        for c in ALPHABET:
            c2 = chr(ord(c) + 1)
            self.bounds[c] = (bisect.bisect(self.words, c),
                              bisect.bisect(self.words, c2))

    def lookup(self, prefix, lo=0, hi=None):
        """See if prefix is in dictionary, as a full word or as a prefix.
        Return two values: the first is the lowest i such that
        words[i].startswith(prefix), or is None; the second is
        True iff prefix itself is in the Wordlist."""
        words = self.words
        i = bisect.bisect_left(words, prefix, lo, hi)
        if i < len(words) and words[i].startswith(prefix): 
            return i, (words[i] == prefix)
        else: 
            return None, False

    def __contains__(self, word): 
        return self.words[bisect.bisect_left(self.words, word)] == word

    def __len__(self): 
        return len(self.words)
    
##_____________________________________________________________________________

class BoggleFinder:
    """A class that allows you to find all the words in a Boggle board. """

    wordlist = None ## A class variable, holding a wordlist

    def __init__(self, board=None):
        if BoggleFinder.wordlist is None:
            BoggleFinder.wordlist = Wordlist("../data/wordlist")
        self.found = {}
        if board:
            self.set_board(board)

    def set_board(self, board=None):
        "Set the board, and find all the words in it."
        if board is None:
            board = random_boggle()
        self.board = board
        self.neighbors = boggle_neighbors(len(board))
        self.found = {}
        for i in range(len(board)):
            lo, hi = self.wordlist.bounds[board[i]]
            self.find(lo, hi, i, [], '')
        return self
            
    def find(self, lo, hi, i, visited, prefix):
        """Looking in square i, find the words that continue the prefix,
        considering the entries in self.wordlist.words[lo:hi], and not
        revisiting the squares in visited."""
        if i in visited: 
            return
        wordpos, is_word = self.wordlist.lookup(prefix, lo, hi)
        if wordpos is not None:
            if is_word: 
                self.found[prefix] = True
            visited.append(i)
            c = self.board[i]
            if c == 'Q': c = 'QU'
            prefix += c
            for j in self.neighbors[i]: 
                self.find(wordpos, hi, j, visited, prefix)
            visited.pop()

    def words(self): 
        "The words found."
        return self.found.keys()

    scores = [0, 0, 0, 0, 1, 2, 3, 5] + [11] * 100

    def score(self):
        "The total score for the words found, according to the rules."
        return sum([self.scores[len(w)] for w in self.words()])

    def __len__(self):
        "The number of words found."
        return len(self.found)

##_____________________________________________________________________________
    
def boggle_hill_climbing(board=None, ntimes=100, print_it=True):
    """Solve inverse Boggle by hill-climbing: find a high-scoring board by
    starting with a random one and changing it."""
    finder = BoggleFinder()
    if board is None:
        board = random_boggle()
    best = len(finder.set_board(board))
    for _ in range(ntimes):
        i, oldc = mutate_boggle(board)
        new = len(finder.set_board(board))
        if new > best:
            best = new
            print best, _, board
        else:
            board[i] = oldc ## Change back
    if print_it:
        print_boggle(board)
    return board, best

def mutate_boggle(board):
    i = random.randrange(len(board))
    oldc = board[i]
    board[i] = random.choice(random.choice(cubes16)) ##random.choice(boyan_best)
    return i, oldc

#______________________________________________________________________________

## Code to compare searchers on various problems.

class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem): 
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None
        
    def successor(self, state):
        "Return a list of (action, state) pairs reachable from this state."
        result =  self.problem.successor(state)
        self.succs += 1; self.states += len(result)
        return result
    
    def goal_test(self, state):
        "Return true if the state is a goal."
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result: 
            self.found = state
        return result
    
    def __getattr__(self, attr):
        if attr in ('succs', 'goal_tests', 'states'):
            return self.__dict__[attr]
        else:
            return getattr(self.problem, attr)

    def __repr__(self):
        return '<%4d/%4d/%4d/%s>' % (self.succs, self.goal_tests,
                                     self.states, str(self.found)[0:4])

def compare_searchers(problems, header, searchers=[breadth_first_tree_search,
                      breadth_first_graph_search, depth_first_graph_search,
                      iterative_deepening_search, depth_limited_search,
                      astar_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p)
        return p
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

def compare_graph_searchers():
    compare_searchers(problems=[GraphProblem('A', 'B', romania),
                                GraphProblem('O', 'N', romania),
                                GraphProblem('Q', 'WA', australia)],
            header=['Searcher', 'Romania(A,B)', 'Romania(O, N)', 'Australia'])

