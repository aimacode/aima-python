"""Natural Language Processing; Chart Parsing and PageRanking (Chapter 22-23)"""

from collections import defaultdict
from utils import weighted_choice
import urllib.request
import re

# ______________________________________________________________________________
# Grammars and Lexicons


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
            prob = float(r[-1][1:-1]) # remove brackets, convert to float
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
            prob = float(r[-1][1:-1]) # remove brackets, convert to float
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
        import random

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
             Rules(  # Grammar for E_0 [Figure 22.4]
                 S='NP VP | S Conjunction S',
                 NP='Pronoun | Name | Noun | Article Noun | Digit Digit | NP PP | NP RelClause',
                 VP='Verb | VP NP | VP Adjective | VP PP | VP Adverb',
                 PP='Preposition NP',
                 RelClause='That VP'),

             Lexicon(  # Lexicon for E_0 [Figure 22.3]
                 Noun="stench | breeze | glitter | nothing | wumpus | pit | pits | gold | east",
                 Verb="is | see | smell | shoot | fell | stinks | go | grab | carry | kill | turn | feel",  # noqa
                 Adjective="right | left | east | south | back | smelly",
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

E_Prob = ProbGrammar('E_Prob', # The Probabilistic Grammar from the notebook
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



E_Chomsky = Grammar('E_Prob_Chomsky', # A Grammar in Chomsky Normal Form
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

E_Prob_Chomsky = ProbGrammar('E_Prob_Chomsky', # A Probabilistic Grammar in CNF
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
# Chart Parsing


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
        self.chart = [[] for i in range(len(words)+1)]
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
                self.add_edge([i, j+1, A, alpha + [(Bb[0], word)], Bb[1:]])

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

def CYK_parse(words, grammar):
    """ [Figure 23.5] """
    # We use 0-based indexing instead of the book's 1-based.
    N = len(words)
    P = defaultdict(float)

    # Insert lexical rules for each word.
    for (i, word) in enumerate(words):
        for (X, p) in grammar.categories[word]:
            P[X, i, 1] = p

    # Combine first and second parts of right-hand sides of rules,
    # from short to long.
    for length in range(2, N+1):
        for start in range(N-length+1):
            for len1 in range(1, length):  # N.B. the book incorrectly has N instead of length
                len2 = length - len1
                for (X, Y, Z, p) in grammar.cnf_rules():
                    P[X, start, length] = max(P[X, start, length],
                                              P[Y, start, len1] * P[Z, start+len1, len2] * p)

    return P


# ______________________________________________________________________________
# Page Ranking

# First entry in list is the base URL, and then following are relative URL pages
examplePagesSet = ["https://en.wikipedia.org/wiki/", "Aesthetics", "Analytic_philosophy",
                   "Ancient_Greek", "Aristotle", "Astrology", "Atheism", "Baruch_Spinoza",
                   "Belief", "Betrand Russell", "Confucius", "Consciousness",
                   "Continental Philosophy", "Dialectic", "Eastern_Philosophy",
                   "Epistemology", "Ethics", "Existentialism", "Friedrich_Nietzsche",
                   "Idealism", "Immanuel_Kant", "List_of_political_philosophers", "Logic",
                   "Metaphysics", "Philosophers", "Philosophy", "Philosophy_of_mind", "Physics",
                   "Plato", "Political_philosophy", "Pythagoras", "Rationalism",
                   "Social_philosophy", "Socrates", "Subjectivity", "Theology",
                   "Truth", "Western_philosophy"]


def loadPageHTML(addressList):
    """Download HTML page content for every URL address passed as argument"""
    contentDict = {}
    for addr in addressList:
        with urllib.request.urlopen(addr) as response:
            raw_html = response.read().decode('utf-8')
            # Strip raw html of unnessecary content. Basically everything that isn't link or text
            html = stripRawHTML(raw_html)
            contentDict[addr] = html
    return contentDict


def initPages(addressList):
    """Create a dictionary of pages from a list of URL addresses"""
    pages = {}
    for addr in addressList:
        pages[addr] = Page(addr)
    return pages


def stripRawHTML(raw_html):
    """Remove the <head> section of the HTML which contains links to stylesheets etc.,
    and remove all other unnessecary HTML"""
    # TODO: Strip more out of the raw html
    return re.sub("<head>.*?</head>", "", raw_html, flags=re.DOTALL)  # remove <head> section


def determineInlinks(page):
    """Given a set of pages that have their outlinks determined, we can fill
    out a page's inlinks by looking through all other page's outlinks"""
    inlinks = []
    for addr, indexPage in pagesIndex.items():
        if page.address == indexPage.address:
            continue
        elif page.address in indexPage.outlinks:
            inlinks.append(addr)
    return inlinks


def findOutlinks(page, handleURLs=None):
    """Search a page's HTML content for URL links to other pages"""
    urls = re.findall(r'href=[\'"]?([^\'" >]+)', pagesContent[page.address])
    if handleURLs:
        urls = handleURLs(urls)
    return urls


def onlyWikipediaURLS(urls):
    """Some example HTML page data is from wikipedia. This function converts
    relative wikipedia links to full wikipedia URLs"""
    wikiURLs = [url for url in urls if url.startswith('/wiki/')]
    return ["https://en.wikipedia.org"+url for url in wikiURLs]


# ______________________________________________________________________________
# HITS Helper Functions

def expand_pages(pages):
    """Adds in every page that links to or is linked from one of
    the relevant pages."""
    expanded = {}
    for addr, page in pages.items():
        if addr not in expanded:
            expanded[addr] = page
        for inlink in page.inlinks:
            if inlink not in expanded:
                expanded[inlink] = pagesIndex[inlink]
        for outlink in page.outlinks:
            if outlink not in expanded:
                expanded[outlink] = pagesIndex[outlink]
    return expanded


def relevant_pages(query):
    """Relevant pages are pages that contain all of the query words. They are obtained by
    intersecting the hit lists of the query words."""
    hit_intersection = {addr for addr in pagesIndex}
    query_words = query.split()
    for query_word in query_words:
        hit_list = set()
        for addr in pagesIndex:
            if query_word.lower() in pagesContent[addr].lower():
                hit_list.add(addr)
        hit_intersection = hit_intersection.intersection(hit_list)
    return {addr: pagesIndex[addr] for addr in hit_intersection}


def normalize(pages):
    """Normalize divides each page's score by the sum of the squares of all
    pages' scores (separately for both the authority and hub scores).
    """
    summed_hub = sum(page.hub**2 for _, page in pages.items())
    summed_auth = sum(page.authority**2 for _, page in pages.items())
    for _, page in pages.items():
        page.hub /= summed_hub**0.5
        page.authority /= summed_auth**0.5


class ConvergenceDetector(object):
    """If the hub and authority values of the pages are no longer changing, we have
    reached a convergence and further iterations will have no effect. This detects convergence
    so that we can stop the HITS algorithm as early as possible."""
    def __init__(self):
        self.hub_history = None
        self.auth_history = None

    def __call__(self):
        return self.detect()

    def detect(self):
        curr_hubs = [page.hub for addr, page in pagesIndex.items()]
        curr_auths = [page.authority for addr, page in pagesIndex.items()]
        if self.hub_history is None:
            self.hub_history, self.auth_history = [], []
        else:
            diffsHub = [abs(x-y) for x, y in zip(curr_hubs, self.hub_history[-1])]
            diffsAuth = [abs(x-y) for x, y in zip(curr_auths, self.auth_history[-1])]
            aveDeltaHub  = sum(diffsHub)/float(len(pagesIndex))
            aveDeltaAuth = sum(diffsAuth)/float(len(pagesIndex))
            if aveDeltaHub < 0.01 and aveDeltaAuth < 0.01:  # may need tweaking
                return True
        if len(self.hub_history) > 2:  # prevent list from getting long
            del self.hub_history[0]
            del self.auth_history[0]
        self.hub_history.append([x for x in curr_hubs])
        self.auth_history.append([x for x in curr_auths])
        return False


def getInlinks(page):
    if not page.inlinks:
        page.inlinks = determineInlinks(page)
    return [addr for addr, p in pagesIndex.items() if addr in page.inlinks]


def getOutlinks(page):
    if not page.outlinks:
        page.outlinks = findOutlinks(page)
    return [addr for addr, p in pagesIndex.items() if addr in page.outlinks]


# ______________________________________________________________________________
# HITS Algorithm

class Page(object):
    def __init__(self, address, inlinks=None, outlinks=None, hub=0, authority=0):
        self.address = address
        self.hub = hub
        self.authority = authority
        self.inlinks = inlinks
        self.outlinks = outlinks


pagesContent = {}  # maps Page relative or absolute URL/location to page's HTML content
pagesIndex = {}
convergence = ConvergenceDetector()  # assign function to variable to mimic pseudocode's syntax


def HITS(query):
    """The HITS algorithm for computing hubs and authorities with respect to a query."""
    pages = expand_pages(relevant_pages(query))
    for p in pages.values():
        p.authority = 1
        p.hub = 1
    while not convergence():
        authority = {p: pages[p].authority for p in pages}
        hub = {p: pages[p].hub for p in pages}
        for p in pages:
            # p.authority ← ∑i Inlinki(p).Hub
            pages[p].authority = sum(hub[x] for x in getInlinks(pages[p]))
            # p.hub ← ∑i Outlinki(p).Authority
            pages[p].hub = sum(authority[x] for x in getOutlinks(pages[p]))
        normalize(pages)
    return pages
