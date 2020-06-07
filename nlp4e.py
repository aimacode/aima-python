"""Natural Language Processing (Chapter 22)"""

from collections import defaultdict
from utils4e import weighted_choice
import copy
import operator
import heapq
from search import Problem


# ______________________________________________________________________________
# 22.2 Grammars


def Rules(**rules):
    """Create a dictionary mapping symbols to alternative sequences.
    >>> Rules(A = "B C | D E")
    {'A': [['B', 'C'], ['D', 'E']]}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = [alt.strip().split() for alt in rhs.split('|')]
    return rules


def Lexicon(**rules):
    """Create a dictionary mapping symbols to alternative words.
    >>> Lexicon(Article = "the | a | an")
    {'Article': ['the', 'a', 'an']}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = [word.strip() for word in rhs.split('|')]
    return rules


class Grammar:

    def __init__(self, name, rules, lexicon):
        """A grammar has a set of rules and a lexicon."""
        self.name = name
        self.rules = rules
        self.lexicon = lexicon
        self.categories = defaultdict(list)
        for lhs in lexicon:
            for word in lexicon[lhs]:
                self.categories[word].append(lhs)

    def rewrites_for(self, cat):
        """Return a sequence of possible rhs's that cat can be rewritten as."""
        return self.rules.get(cat, ())

    def isa(self, word, cat):
        """Return True iff word is of category cat"""
        return cat in self.categories[word]

    def cnf_rules(self):
        """Returns the tuple (X, Y, Z) for rules in the form:
        X -> Y Z"""
        cnf = []
        for X, rules in self.rules.items():
            for (Y, Z) in rules:
                cnf.append((X, Y, Z))

        return cnf

    def generate_random(self, S='S'):
        """Replace each token in S by a random entry in grammar (recursively)."""
        import random

        def rewrite(tokens, into):
            for token in tokens:
                if token in self.rules:
                    rewrite(random.choice(self.rules[token]), into)
                elif token in self.lexicon:
                    into.append(random.choice(self.lexicon[token]))
                else:
                    into.append(token)
            return into

        return ' '.join(rewrite(S.split(), []))

    def __repr__(self):
        return '<Grammar {}>'.format(self.name)


def ProbRules(**rules):
    """Create a dictionary mapping symbols to alternative sequences,
    with probabilities.
    >>> ProbRules(A = "B C [0.3] | D E [0.7]")
    {'A': [(['B', 'C'], 0.3), (['D', 'E'], 0.7)]}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = []
        rhs_separate = [alt.strip().split() for alt in rhs.split('|')]
        for r in rhs_separate:
            prob = float(r[-1][1:-1])  # remove brackets, convert to float
            rhs_rule = (r[:-1], prob)
            rules[lhs].append(rhs_rule)

    return rules


def ProbLexicon(**rules):
    """Create a dictionary mapping symbols to alternative words,
    with probabilities.
    >>> ProbLexicon(Article = "the [0.5] | a [0.25] | an [0.25]")
    {'Article': [('the', 0.5), ('a', 0.25), ('an', 0.25)]}
    """
    for (lhs, rhs) in rules.items():
        rules[lhs] = []
        rhs_separate = [word.strip().split() for word in rhs.split('|')]
        for r in rhs_separate:
            prob = float(r[-1][1:-1])  # remove brackets, convert to float
            word = r[:-1][0]
            rhs_rule = (word, prob)
            rules[lhs].append(rhs_rule)

    return rules


class ProbGrammar:

    def __init__(self, name, rules, lexicon):
        """A grammar has a set of rules and a lexicon.
        Each rule has a probability."""
        self.name = name
        self.rules = rules
        self.lexicon = lexicon
        self.categories = defaultdict(list)

        for lhs in lexicon:
            for word, prob in lexicon[lhs]:
                self.categories[word].append((lhs, prob))

    def rewrites_for(self, cat):
        """Return a sequence of possible rhs's that cat can be rewritten as."""
        return self.rules.get(cat, ())

    def isa(self, word, cat):
        """Return True iff word is of category cat"""
        return cat in [c for c, _ in self.categories[word]]

    def cnf_rules(self):
        """Returns the tuple (X, Y, Z, p) for rules in the form:
        X -> Y Z [p]"""
        cnf = []
        for X, rules in self.rules.items():
            for (Y, Z), p in rules:
                cnf.append((X, Y, Z, p))

        return cnf

    def generate_random(self, S='S'):
        """Replace each token in S by a random entry in grammar (recursively).
        Returns a tuple of (sentence, probability)."""

        def rewrite(tokens, into):
            for token in tokens:
                if token in self.rules:
                    non_terminal, prob = weighted_choice(self.rules[token])
                    into[1] *= prob
                    rewrite(non_terminal, into)
                elif token in self.lexicon:
                    terminal, prob = weighted_choice(self.lexicon[token])
                    into[0].append(terminal)
                    into[1] *= prob
                else:
                    into[0].append(token)
            return into

        rewritten_as, prob = rewrite(S.split(), [[], 1])
        return (' '.join(rewritten_as), prob)

    def __repr__(self):
        return '<Grammar {}>'.format(self.name)


E0 = Grammar('E0',
             Rules(  # Grammar for E_0 [Figure 22.2]
                 S='NP VP | S Conjunction S',
                 NP='Pronoun | Name | Noun | Article Noun | Digit Digit | NP PP | NP RelClause',
                 VP='Verb | VP NP | VP Adjective | VP PP | VP Adverb',
                 PP='Preposition NP',
                 RelClause='That VP'),

             Lexicon(  # Lexicon for E_0 [Figure 22.3]
                 Noun="stench | breeze | glitter | nothing | wumpus | pit | pits | gold | east",
                 Verb="is | see | smell | shoot | fell | stinks | go | grab | carry | kill | turn | feel",  # noqa
                 Adjective="right | left | east | south | back | smelly | dead",
                 Adverb="here | there | nearby | ahead | right | left | east | south | back",
                 Pronoun="me | you | I | it",
                 Name="John | Mary | Boston | Aristotle",
                 Article="the | a | an",
                 Preposition="to | in | on | near",
                 Conjunction="and | or | but",
                 Digit="0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9",
                 That="that"
             ))

E_ = Grammar('E_',  # Trivial Grammar and lexicon for testing
             Rules(
                 S='NP VP',
                 NP='Art N | Pronoun',
                 VP='V NP'),

             Lexicon(
                 Art='the | a',
                 N='man | woman | table | shoelace | saw',
                 Pronoun='I | you | it',
                 V='saw | liked | feel'
             ))

E_NP_ = Grammar('E_NP_',  # Another Trivial Grammar for testing
                Rules(NP='Adj NP | N'),
                Lexicon(Adj='happy | handsome | hairy',
                        N='man'))

E_Prob = ProbGrammar('E_Prob',  # The Probabilistic Grammar from the notebook
                     ProbRules(
                         S="NP VP [0.6] | S Conjunction S [0.4]",
                         NP="Pronoun [0.2] | Name [0.05] | Noun [0.2] | Article Noun [0.15] \
                             | Article Adjs Noun [0.1] | Digit [0.05] | NP PP [0.15] | NP RelClause [0.1]",
                         VP="Verb [0.3] | VP NP [0.2] | VP Adjective [0.25] | VP PP [0.15] | VP Adverb [0.1]",
                         Adjs="Adjective [0.5] | Adjective Adjs [0.5]",
                         PP="Preposition NP [1]",
                         RelClause="RelPro VP [1]"
                     ),
                     ProbLexicon(
                         Verb="is [0.5] | say [0.3] | are [0.2]",
                         Noun="robot [0.4] | sheep [0.4] | fence [0.2]",
                         Adjective="good [0.5] | new [0.2] | sad [0.3]",
                         Adverb="here [0.6] | lightly [0.1] | now [0.3]",
                         Pronoun="me [0.3] | you [0.4] | he [0.3]",
                         RelPro="that [0.5] | who [0.3] | which [0.2]",
                         Name="john [0.4] | mary [0.4] | peter [0.2]",
                         Article="the [0.5] | a [0.25] | an [0.25]",
                         Preposition="to [0.4] | in [0.3] | at [0.3]",
                         Conjunction="and [0.5] | or [0.2] | but [0.3]",
                         Digit="0 [0.35] | 1 [0.35] | 2 [0.3]"
                     ))

E_Chomsky = Grammar('E_Prob_Chomsky',  # A Grammar in Chomsky Normal Form
                    Rules(
                        S='NP VP',
                        NP='Article Noun | Adjective Noun',
                        VP='Verb NP | Verb Adjective',
                    ),
                    Lexicon(
                        Article='the | a | an',
                        Noun='robot | sheep | fence',
                        Adjective='good | new | sad',
                        Verb='is | say | are'
                    ))

E_Prob_Chomsky = ProbGrammar('E_Prob_Chomsky',  # A Probabilistic Grammar in CNF
                             ProbRules(
                                 S='NP VP [1]',
                                 NP='Article Noun [0.6] | Adjective Noun [0.4]',
                                 VP='Verb NP [0.5] | Verb Adjective [0.5]',
                             ),
                             ProbLexicon(
                                 Article='the [0.5] | a [0.25] | an [0.25]',
                                 Noun='robot [0.4] | sheep [0.4] | fence [0.2]',
                                 Adjective='good [0.5] | new [0.2] | sad [0.3]',
                                 Verb='is [0.5] | say [0.3] | are [0.2]'
                             ))
E_Prob_Chomsky_ = ProbGrammar('E_Prob_Chomsky_',
                              ProbRules(
                                  S='NP VP [1]',
                                  NP='NP PP [0.4] | Noun Verb [0.6]',
                                  PP='Preposition NP [1]',
                                  VP='Verb NP [0.7] | VP PP [0.3]',
                              ),
                              ProbLexicon(
                                  Noun='astronomers [0.18] | eyes [0.32] | stars [0.32] | telescopes [0.18]',
                                  Verb='saw [0.5] | \'\' [0.5]',
                                  Preposition='with [1]'
                              ))


# ______________________________________________________________________________
# 22.3 Parsing


class Chart:
    """Class for parsing sentences using a chart data structure.
    >>> chart = Chart(E0)
    >>> len(chart.parses('the stench is in 2 2'))
    1
    """

    def __init__(self, grammar, trace=False):
        """A datastructure for parsing a string; and methods to do the parse.
        self.chart[i] holds the edges that end just before the i'th word.
        Edges are 5-element lists of [start, end, lhs, [found], [expects]]."""
        self.grammar = grammar
        self.trace = trace

    def parses(self, words, S='S'):
        """Return a list of parses; words can be a list or string."""
        if isinstance(words, str):
            words = words.split()
        self.parse(words, S)
        # Return all the parses that span the whole input
        # 'span the whole input' => begin at 0, end at len(words)
        return [[i, j, S, found, []]
                for (i, j, lhs, found, expects) in self.chart[len(words)]
                # assert j == len(words)
                if i == 0 and lhs == S and expects == []]

    def parse(self, words, S='S'):
        """Parse a list of words; according to the grammar.
        Leave results in the chart."""
        self.chart = [[] for i in range(len(words) + 1)]
        self.add_edge([0, 0, 'S_', [], [S]])
        for i in range(len(words)):
            self.scanner(i, words[i])
        return self.chart

    def add_edge(self, edge):
        """Add edge to chart, and see if it extends or predicts another edge."""
        start, end, lhs, found, expects = edge
        if edge not in self.chart[end]:
            self.chart[end].append(edge)
            if self.trace:
                print('Chart: added {}'.format(edge))
            if not expects:
                self.extender(edge)
            else:
                self.predictor(edge)

    def scanner(self, j, word):
        """For each edge expecting a word of this category here, extend the edge."""
        for (i, j, A, alpha, Bb) in self.chart[j]:
            if Bb and self.grammar.isa(word, Bb[0]):
                self.add_edge([i, j + 1, A, alpha + [(Bb[0], word)], Bb[1:]])

    def predictor(self, edge):
        """Add to chart any rules for B that could help extend this edge."""
        (i, j, A, alpha, Bb) = edge
        B = Bb[0]
        if B in self.grammar.rules:
            for rhs in self.grammar.rewrites_for(B):
                self.add_edge([j, j, B, [], rhs])

    def extender(self, edge):
        """See what edges can be extended by this edge."""
        (j, k, B, _, _) = edge
        for (i, j, A, alpha, B1b) in self.chart[j]:
            if B1b and B == B1b[0]:
                self.add_edge([i, k, A, alpha + [edge], B1b[1:]])


# ______________________________________________________________________________
# CYK Parsing


class Tree:
    def __init__(self, root, *args):
        self.root = root
        self.leaves = [leaf for leaf in args]


def CYK_parse(words, grammar):
    """ [Figure 22.6] """
    # We use 0-based indexing instead of the book's 1-based.
    P = defaultdict(float)
    T = defaultdict(Tree)

    # Insert lexical categories for each word.
    for (i, word) in enumerate(words):
        for (X, p) in grammar.categories[word]:
            P[X, i, i] = p
            T[X, i, i] = Tree(X, word)

    # Construct X(i:k) from Y(i:j) and Z(j+1:k), shortest span first
    for i, j, k in subspan(len(words)):
        for (X, Y, Z, p) in grammar.cnf_rules():
            PYZ = P[Y, i, j] * P[Z, j + 1, k] * p
            if PYZ > P[X, i, k]:
                P[X, i, k] = PYZ
                T[X, i, k] = Tree(X, T[Y, i, j], T[Z, j + 1, k])

    return T


def subspan(N):
    """returns all tuple(i, j, k) covering a span (i, k) with i <= j < k"""
    for length in range(2, N + 1):
        for i in range(1, N + 2 - length):
            k = i + length - 1
            for j in range(i, k):
                yield (i, j, k)


# using search algorithms in the searching part


class TextParsingProblem(Problem):
    def __init__(self, initial, grammar, goal='S'):
        """
        :param initial: the initial state of words in a list.
        :param grammar: a grammar object
        :param goal: the goal state, usually S
        """
        super(TextParsingProblem, self).__init__(initial, goal)
        self.grammar = grammar
        self.combinations = defaultdict(list)  # article combinations
        # backward lookup of rules
        for rule in grammar.rules:
            for comb in grammar.rules[rule]:
                self.combinations[' '.join(comb)].append(rule)

    def actions(self, state):
        actions = []
        categories = self.grammar.categories
        # first change each word to the article of its category
        for i in range(len(state)):
            word = state[i]
            if word in categories:
                for X in categories[word]:
                    state[i] = X
                    actions.append(copy.copy(state))
                    state[i] = word
        # if all words are replaced by articles, replace combinations of articles by inferring rules.
        if not actions:
            for start in range(len(state)):
                for end in range(start, len(state) + 1):
                    # try combinations between (start, end)
                    articles = ' '.join(state[start:end])
                    for c in self.combinations[articles]:
                        actions.append(state[:start] + [c] + state[end:])
        return actions

    def result(self, state, action):
        return action

    def h(self, state):
        # heuristic function
        return len(state)


def astar_search_parsing(words, gramma):
    """bottom-up parsing using A* search to find whether a list of words is a sentence"""
    # init the problem
    problem = TextParsingProblem(words, gramma, 'S')
    state = problem.initial
    # init the searching frontier
    frontier = [(len(state) + problem.h(state), state)]
    heapq.heapify(frontier)

    while frontier:
        # search the frontier node with lowest cost first
        cost, state = heapq.heappop(frontier)
        actions = problem.actions(state)
        for action in actions:
            new_state = problem.result(state, action)
            # update the new frontier node to the frontier
            if new_state == [problem.goal]:
                return problem.goal
            if new_state != state:
                heapq.heappush(frontier, (len(new_state) + problem.h(new_state), new_state))
    return False


def beam_search_parsing(words, gramma, b=3):
    """bottom-up text parsing using beam search"""
    # init problem
    problem = TextParsingProblem(words, gramma, 'S')
    # init frontier
    frontier = [(len(problem.initial), problem.initial)]
    heapq.heapify(frontier)

    # explore the current frontier and keep b new states with lowest cost
    def explore(frontier):
        new_frontier = []
        for cost, state in frontier:
            # expand the possible children states of current state
            if not problem.goal_test(' '.join(state)):
                actions = problem.actions(state)
                for action in actions:
                    new_state = problem.result(state, action)
                    if [len(new_state), new_state] not in new_frontier and new_state != state:
                        new_frontier.append([len(new_state), new_state])
            else:
                return problem.goal
        heapq.heapify(new_frontier)
        # only keep b states
        return heapq.nsmallest(b, new_frontier)

    while frontier:
        frontier = explore(frontier)
        if frontier == problem.goal:
            return frontier
    return False


# ______________________________________________________________________________
# 22.4 Augmented Grammar


g = Grammar("arithmetic_expression",  # A Grammar of Arithmetic Expression
            rules={
                'Number_0': 'Digit_0', 'Number_1': 'Digit_1', 'Number_2': 'Digit_2',
                'Number_10': 'Number_1 Digit_0', 'Number_11': 'Number_1 Digit_1',
                'Number_100': 'Number_10 Digit_0',
                'Exp_5': ['Number_5', '( Exp_5 )', 'Exp_1, Operator_+ Exp_4', 'Exp_2, Operator_+ Exp_3',
                          'Exp_0, Operator_+ Exp_5', 'Exp_3, Operator_+ Exp_2', 'Exp_4, Operator_+ Exp_1',
                          'Exp_5, Operator_+ Exp_0', 'Exp_1, Operator_* Exp_5'],  # more possible combinations
                'Operator_+': operator.add, 'Operator_-': operator.sub, 'Operator_*': operator.mul,
                'Operator_/': operator.truediv,
                'Digit_0': 0, 'Digit_1': 1, 'Digit_2': 2, 'Digit_3': 3, 'Digit_4': 4
            },
            lexicon={})

g = Grammar("Ali loves Bob",  # A example grammer of Ali loves Bob example
            rules={
                "S_loves_ali_bob": "NP_ali, VP_x_loves_x_bob", "S_loves_bob_ali": "NP_bob, VP_x_loves_x_ali",
                "VP_x_loves_x_bob": "Verb_xy_loves_xy NP_bob", "VP_x_loves_x_ali": "Verb_xy_loves_xy NP_ali",
                "NP_bob": "Name_bob", "NP_ali": "Name_ali"
            },
            lexicon={
                "Name_ali": "Ali", "Name_bob": "Bob", "Verb_xy_loves_xy": "loves"
            })
