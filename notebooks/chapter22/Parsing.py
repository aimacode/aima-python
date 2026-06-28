# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os, sys
sys.path = [os.path.abspath('../../')] + sys.path  # make the aima package importable
from aima.nlp import *
from aima import nlp
from aima.notebook_utils import psource


# %% [markdown]
# # Parsing
#
# ## Overview
#
# Syntactic analysis (or **parsing**) of a sentence is the process of uncovering the phrase structure of the sentence according to the rules of grammar. 
#
# There are two main approaches to parsing. *Top-down*, start with the starting symbol and build a parse tree with the given words as its leaves, and *bottom-up*, where we start from the given words and build a tree that has the starting symbol as its root. Both approaches involve "guessing" ahead, so it may take longer to parse a sentence (the wrong guess mean a lot of backtracking). Thankfully, a lot of effort is spent in analyzing already analyzed substrings, so we can follow a dynamic programming approach to store and reuse these parses instead of recomputing them. 
#
# In dynamic programming, we use a data structure known as a chart, thus the algorithms parsing a chart is called **chart parsing**. We will cover several different chart parsing algorithms.

# %% [markdown]
# ## Chart Parsing
#
# ### Overview
#
# The chart parsing algorithm is a general form of the following algorithms. Given a non-probabilistic grammar and a sentence, this algorithm builds a parse tree in a top-down manner, with the words of the sentence as the leaves. It works with a dynamic programming approach, building a chart to store parses for substrings so that it doesn't have to analyze them again (just like the CYK algorithm). Each non-terminal, starting from S, gets replaced by its right-hand side rules in the chart until we end up with the correct parses.
#
# ### Implementation
#
# A parse is in the form `[start, end, non-terminal, sub-tree, expected-transformation]`, where `sub-tree` is a tree with the corresponding `non-terminal` as its root and `expected-transformation` is a right-hand side rule of the `non-terminal`.
#
# The chart parsing is implemented in a class, `Chart`. It is initialized with grammar and can return the list of all the parses of a sentence with the `parses` function.
#
# The chart is a list of lists. The lists correspond to the lengths of substrings (including the empty string), from start to finish. When we say 'a point in the chart', we refer to a list of a certain length.
#
# A quick rundown of the class functions:

# %% [markdown]
# * `parses`: Returns a list of parses for a given sentence. If the sentence can't be parsed, it will return an empty list. Initializes the process by calling `parse` from the starting symbol.
#
#
# * `parse`: Parses the list of words and builds the chart.
#
#
# * `add_edge`: Adds another edge to the chart at a given point. Also, examines whether the edge extends or predicts another edge. If the edge itself is not expecting a transformation, it will extend other edges and it will predict edges otherwise.
#
#
# * `scanner`: Given a word and a point in the chart, it extends edges that were expecting a transformation that can result in the given word. For example, if the word 'the' is an 'Article' and we are examining two edges at a chart's point, with one expecting an 'Article' and the other a 'Verb', the first one will be extended while the second one will not.
#
#
# * `predictor`: If an edge can't extend other edges (because it is expecting a transformation itself), we will add to the chart rules/transformations that can help extend the edge. The new edges come from the right-hand side of the expected transformation's rules. For example, if an edge is expecting the transformation 'Adjective Noun', we will add to the chart an edge for each right-hand side rule of the non-terminal 'Adjective'.
#
#
# * `extender`: Extends edges given an edge (called `E`). If `E`'s non-terminal is the same as the expected transformation of another edge (let's call it `A`), add to the chart a new edge with the non-terminal of `A` and the transformations of `A` minus the non-terminal that matched with `E`'s non-terminal. For example, if an edge `E` has 'Article' as its non-terminal and is expecting no transformation, we need to see what edges it can extend. Let's examine the edge `N`. This expects a transformation of 'Noun Verb'. 'Noun' does not match with 'Article', so we move on. Another edge, `A`, expects a transformation of 'Article Noun' and has a non-terminal of 'NP'. We have a match! A new edge will be added with 'NP' as its non-terminal (the non-terminal of `A`) and 'Noun' as the expected transformation (the rest of the expected transformation of `A`).
#
# You can view the source code by running the cell below:

# %%
psource(Chart)

# %% [markdown]
# ### Example
#
# We will use the grammar `E0` to parse the sentence "the stench is in 2 2".
#
# First, we need to build a `Chart` object:

# %%
chart = Chart(E0)

# %% [markdown]
# And then we simply call the `parses` function:

# %%
print(chart.parses('the stench is in 2 2'))

# %% [markdown]
# You can see which edges get added by setting the optional initialization argument `trace` to true.

# %%
chart_trace = Chart(nlp.E0, trace=True)
chart_trace.parses('the stench is in 2 2')

# %% [markdown]
# Let's try and parse a sentence that is not recognized by the grammar:

# %%
print(chart.parses('the stench 2 2'))

# %% [markdown]
# An empty list was returned.

# %% [markdown]
# ## CYK Parse
#
# The *CYK Parsing Algorithm* (named after its inventors, Cocke, Younger, and Kasami) utilizes dynamic programming to parse sentences of grammar in *Chomsky Normal Form*.
#
# The CYK algorithm returns an *M x N x N* array (named *P*), where *N* is the number of words in the sentence and *M* the number of non-terminal symbols in the grammar. Each element in this array shows the probability of a substring being transformed from a particular non-terminal. To find the most probable parse of the sentence, a search in the resulting array is required. Search heuristic algorithms work well in this space, and we can derive the heuristics from the properties of the grammar.
#
# The algorithm in short works like this: There is an external loop that determines the length of the substring. Then the algorithm loops through the words in the sentence. For each word, it again loops through all the words to its right up to the first-loop length. The substring will work on in this iteration is the words from the second-loop word with the first-loop length. Finally, it loops through all the rules in the grammar and updates the substring's probability for each right-hand side non-terminal.

# %% [markdown]
# ### Implementation
#
# The implementation takes as input a list of words and a probabilistic grammar (from the `ProbGrammar` class detailed above) in CNF and returns the table/dictionary *P*. An item's key in *P* is a tuple in the form `(Non-terminal, the start of a substring, length of substring)`, and the value is a `Tree` object. The `Tree` data structure has two attributes: `root` and `leaves`. `root` stores the value of current tree node and `leaves` is a list of children nodes which may be terminal states(words in the sentence) or a sub tree.
#
# For example, for the sentence "the monkey is dancing" and the substring "the monkey" an item can be `('NP', 0, 2): <Tree object>`, which means the first two words (the substring from index 0 and length 2) can be parse to a `NP` and the detailed operations are recorded by a `Tree` object.
#
# Before we continue, you can take a look at the source code by running the cell below:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.nlp import *
from aima import nlp
from aima.notebook_utils import psource

# %%
psource(CYK_parse)

# %% [markdown]
# When updating the probability of a substring, we pick the max of its current one and the probability of the substring broken into two parts: one from the second-loop word with third-loop length, and the other from the first part's end to the remainder of the first-loop length.
#
# ### Example
#
# Let's build a probabilistic grammar in CNF:

# %%
E_Prob_Chomsky = ProbGrammar("E_Prob_Chomsky", # A Probabilistic Grammar in CNF
                             ProbRules(
                                S = "NP VP [1]",
                                NP = "Article Noun [0.6] | Adjective Noun [0.4]",
                                VP = "Verb NP [0.5] | Verb Adjective [0.5]",
                             ),
                             ProbLexicon(
                                Article = "the [0.5] | a [0.25] | an [0.25]",
                                Noun = "robot [0.4] | sheep [0.4] | fence [0.2]",
                                Adjective = "good [0.5] | new [0.2] | sad [0.3]",
                                Verb = "is [0.5] | say [0.3] | are [0.2]"
                             ))

# %% [markdown]
# Now let's see the probabilities table for the sentence "the robot is good":

# %%
words = ['the', 'robot', 'is', 'good']
grammar = E_Prob_Chomsky

P = CYK_parse(words, grammar)
print(P)

# %% [markdown]
# A `defaultdict` object is returned (`defaultdict` is basically a dictionary but with a default value/type). Keys are tuples in the form mentioned above and the values are the corresponding parse trees which demonstrates how the sentence will be parsed. Let's check the details of each parsing:

# %%
parses = {k: p for k, p in P.items() if p}

print(parses)

# %% [markdown]
# Please note that each item in the returned dict represents a parsing strategy. For instance, `('Article', 0, 0): ['the']` means parsing the article at position 0 from the word `the`. For the key `'VP', 2, 3`, it is mapped to another `Tree` which means this is a nested parsing step. If we print this item in detail: 

# %%
# CYK_parse returns a probability table P[(symbol, start, length)] -> probability;
# show where the grammar can parse a VP (verb phrase) and with what probability
print({k: p for k, p in P.items() if k[0] == 'VP' and p})

# %% [markdown]
# So we can interpret this step as parsing the word at index 2 and 3 together('is' and 'good') as a verh phrase.

# %% [markdown]
# ## A-star Parsing
#
# The CYK algorithm uses space of $O(n^2m)$ for the P and T tables, where n is the number of words in the sentence, and m is the number of nonterminal symbols in the grammar and takes time $O(n^3m)$. This is the best algorithm if we want to ﬁnd the best parse and works for all possible context-free grammars. But actually, we only want to parse natural languages, not all possible grammars, which allows us to apply more efficient algorithms.
#
# By applying a-start search, we are using the state-space search and we can get $O(n)$ running time. In this situation, each state is a list of items (words or categories), the start state is a list of words, and a goal state is the single item S. 
#
# In our code, we implemented a demonstration of `astar_search_parsing` which deals with the text parsing problem. By specifying different `words` and `gramma`, we can use this searching strategy to deal with different text parsing problems. The algorithm returns a boolean telling whether the input words is a sentence under the given grammar.
#
# For detailed implementation, please execute the following block:

# %%
psource(astar_search_parsing)

# %% [markdown]
# ### Example
#
# Now let's try "the wumpus is dead" example. First we need to define the grammar and words in the sentence.

# %%
grammar = E0
words = ['the', 'wumpus', 'is', 'dead']

# %%
astar_search_parsing(words, grammar)

# %% [markdown]
# The algorithm returns a 'S' which means it treats the inputs as a sentence. If we change the order of words to make it unreadable:

# %%
words_swaped = ["the", "is", "wupus", "dead"]
astar_search_parsing(words_swaped, grammar)

# %% [markdown]
# Then the algorithm asserts that out words cannot be a sentence.

# %% [markdown]
# ## Beam Search Parsing
#
# In the beam searching algorithm, we still treat the text parsing problem as a state-space searching algorithm. when using beam search, we consider only the b most probable alternative parses. This means we are not guaranteed to ﬁnd the parse with the highest probability, but (with a careful implementation) the parser can operate in $O(n)$ time and still ﬁnds the best parse most of the time. A beam search parser with b = 1 is called a **deterministic parser**.
#
# ### Implementation
#
# In the beam search, we maintain a `frontier` which is a priority queue keep tracking of the current frontier of searching. In each step, we explore all the examples in `frontier` and saves the best n examples as the frontier of the exploration of the next step.
#
# For detailed implementation, please view with the following code:

# %%
psource(beam_search_parsing)

# %% [markdown]
# ### Example
#
# Let's try both the positive and negative wumpus example on this algorithm:

# %%
beam_search_parsing(words, grammar)

# %%
beam_search_parsing(words_swaped, grammar)
