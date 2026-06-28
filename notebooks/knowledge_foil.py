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
# %run bootstrap.ipynb

# %% [markdown]
# # KNOWLEDGE
#
# The [knowledge](https://github.com/aimacode/aima-python/blob/master/knowledge.py) module covers **Chapter 19: Knowledge in Learning** from Stuart Russel's and Peter Norvig's book *Artificial Intelligence: A Modern Approach*.
#
# Execute the cell below to get started.

# %%
from aima.knowledge import *
from aima.notebook_utils import psource

# %% [markdown]
# ## CONTENTS
#
# * Overview
# * Inductive Logic Programming (FOIL)

# %% [markdown]
# ## OVERVIEW
#
# Like the [learning module](https://github.com/aimacode/aima-python/blob/master/learning.ipynb), this chapter focuses on methods for generating a model/hypothesis for a domain; however, unlike the learning chapter, here we use prior knowledge to help us learn from new experiences and to find a proper hypothesis.
#
# ### First-Order Logic
#
# Usually knowledge in this field is represented as **first-order logic**, a type of logic that uses variables and quantifiers in logical sentences. Hypotheses are represented by logical sentences with variables, while examples are logical sentences with set values instead of variables. The goal is to assign a value to a special first-order logic predicate, called **goal predicate**, for new examples given a hypothesis. We learn this hypothesis by infering knowledge from some given examples.
#
# ### Representation
#
# In this module, we use dictionaries to represent examples, with keys being the attribute names and values being the corresponding example values. Examples also have an extra boolean field, 'GOAL', for the goal predicate. A hypothesis is represented as a list of dictionaries. Each dictionary in that list represents a disjunction. Inside these dictionaries/disjunctions we have conjunctions.
#
# For example, say we want to predict if an animal (cat or dog) will take an umbrella given whether or not it rains or the animal wears a coat. The goal value is 'take an umbrella' and is denoted by the key 'GOAL'. An example:
#
# `{'Species': 'Cat', 'Coat': 'Yes', 'Rain': 'Yes', 'GOAL': True}`
#
# A hypothesis can be the following:
#
# `[{'Species': 'Cat'}]`
#
# which means an animal will take an umbrella if and only if it is a cat.
#
# ### Consistency
#
# We say that an example `e` is **consistent** with an hypothesis `h` if the assignment from the hypothesis for `e` is the same as `e['GOAL']`. If the above example and hypothesis are `e` and `h` respectively, then `e` is consistent with `h` since `e['Species'] == 'Cat'`. For `e = {'Species': 'Dog', 'Coat': 'Yes', 'Rain': 'Yes', 'GOAL': True}`, the example is no longer consistent with `h`, since the value assigned to `e` is *False* while `e['GOAL']` is *True*.

# %% [markdown]
# # Inductive Logic Programming (FOIL)
#
# Inductive logic programming (ILP) combines inductive methods with the power of first-order representations, concentrating in particular on the representation of hypotheses as logic programs. The general knowledge-based induction problem is to solve the entailment constraint: <br> <br>
# $ Background ∧ Hypothesis ∧ Descriptions \vDash Classifications $
#
# for the __unknown__ $Hypothesis$, given the $Background$ knowledge described by $Descriptions$ and $Classifications$.
#
#
#
# The first approach to ILP works by starting with a very general rule and gradually specializing
# it so that it fits the data. <br> 
# This is essentially what happens in decision-tree learning, where a
# decision tree is gradually grown until it is consistent with the observations. <br> To do ILP we
# use first-order literals instead of attributes, and the $Hypothesis$ is a set of clauses (set of first order rules, where each rule is similar to a Horn clause) instead of a decision tree. <br>
#
#
# The FOIL algorithm learns new rules, one at a time, in order to cover all given positive and negative examples. <br>
# More precicely, FOIL contains an inner and an outer while loop. <br>
# -  __outer loop__: <font color='blue'>(function __foil()__) </font>  add rules until all positive examples are covered. <br>
#    (each rule is a conjuction of literals, which are chosen inside the inner loop)
#    
#    
# -  __inner loop__: <font color ='blue'>(function __new_clause()__) </font>  add new literals until all negative examples are covered, and some positive examples are covered. <br>
#    -  In each iteration, we select/add the most promising literal, according to an estimate of its utility. <font color ='blue'>(function __new_literal()__) </font> <br>
#    
#    -  The evaluation function to estimate utility of adding literal $L$ to a set of rules $R$ is <font color ='blue'>(function __gain()__) </font> : 
#    
#    $$ FoilGain(L,R) = t \big( \log_2{\frac{p_1}{p_1+n_1}} - \log_2{\frac{p_0}{p_0+n_0}} \big) $$
#       where: 
#       
#       $p_0: \text{is the number of positive bindings of rule R } \\ n_0: \text{is the number of negative bindings of R} \\ p_1: \text{is the is the number of positive bindings of rule R'}\\ n_0: \text{is the number of negative bindings of R'}\\ t: \text{is the number of positive bindings of rule R that are still covered after adding literal L to R}$
#    
#    - Calculate the extended examples for the chosen literal <font color ='blue'>(function __extend_example()__) </font> <br>
#         (the set of examples created by extending example with each possible constant value for each new variable in literal)
#    
# -  Finally, the algorithm returns a disjunction of first order rules (= conjuction of literals)
#
#

# %%
psource(FOILContainer)

# %% [markdown]
# ### Example Family 
# Suppose we have the following family relations:
# <br>
# ![title](images/knowledge_foil_family.png)
# <br>
# Given some positive and negative examples of the relation 'Parent(x,y)', we want to find a set of rules that satisfies all the examples. <br>
#
# A definition of Parent is $Parent(x,y) \Leftrightarrow Mother(x,y) \lor Father(x,y)$, which is the result that we expect from the algorithm. 

# %%
A, B, C, D, E, F, G, H, I, x, y, z = map(expr, 'ABCDEFGHIxyz')

# %%
small_family = FOILContainer([expr("Mother(Anne, Peter)"),
                               expr("Mother(Anne, Zara)"),
                               expr("Mother(Sarah, Beatrice)"),
                               expr("Mother(Sarah, Eugenie)"),
                               expr("Father(Mark, Peter)"),
                               expr("Father(Mark, Zara)"),
                               expr("Father(Andrew, Beatrice)"),
                               expr("Father(Andrew, Eugenie)"),
                               expr("Father(Philip, Anne)"),
                               expr("Father(Philip, Andrew)"),
                               expr("Mother(Elizabeth, Anne)"),
                               expr("Mother(Elizabeth, Andrew)"),
                               expr("Male(Philip)"),
                               expr("Male(Mark)"),
                               expr("Male(Andrew)"),
                               expr("Male(Peter)"),
                               expr("Female(Elizabeth)"),
                               expr("Female(Anne)"),
                               expr("Female(Sarah)"),
                               expr("Female(Zara)"),
                               expr("Female(Beatrice)"),
                               expr("Female(Eugenie)"),
])

target = expr('Parent(x, y)')

examples_pos = [{x: expr('Elizabeth'), y: expr('Anne')},
                {x: expr('Elizabeth'), y: expr('Andrew')},
                {x: expr('Philip'), y: expr('Anne')},
                {x: expr('Philip'), y: expr('Andrew')},
                {x: expr('Anne'), y: expr('Peter')},
                {x: expr('Anne'), y: expr('Zara')},
                {x: expr('Mark'), y: expr('Peter')},
                {x: expr('Mark'), y: expr('Zara')},
                {x: expr('Andrew'), y: expr('Beatrice')},
                {x: expr('Andrew'), y: expr('Eugenie')},
                {x: expr('Sarah'), y: expr('Beatrice')},
                {x: expr('Sarah'), y: expr('Eugenie')}]
examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                {x: expr('Beatrice'), y: expr('Eugenie')},
                {x: expr('Mark'), y: expr('Elizabeth')},
                {x: expr('Beatrice'), y: expr('Philip')}]

# %%
# run the FOIL algorithm 
clauses = small_family.foil([examples_pos, examples_neg], target)
print (clauses)


# %% [markdown]
# Indeed the algorithm returned the rule: 
# <br>$Parent(x,y) \Leftrightarrow Mother(x,y) \lor Father(x,y)$

# %% [markdown]
# Suppose that we have some positive and negative results for the relation 'GrandParent(x,y)' and we want to find a set of rules that satisfies the examples. <br>
# One possible set of rules for the relation $Grandparent(x,y)$ could be: <br>
# ![title](images/knowledge_foil_grandparent.png)
# <br>
# Or, if $Background$ included the sentence $Parent(x,y) \Leftrightarrow [Mother(x,y) \lor Father(x,y)]$ then:  
#
# $$Grandparent(x,y) \Leftrightarrow \exists \: z \quad  Parent(x,z) \land Parent(z,y)$$
#

# %%
target = expr('Grandparent(x, y)')

examples_pos = [{x: expr('Elizabeth'), y: expr('Peter')},
                {x: expr('Elizabeth'), y: expr('Zara')},
                {x: expr('Elizabeth'), y: expr('Beatrice')},
                {x: expr('Elizabeth'), y: expr('Eugenie')},
                {x: expr('Philip'), y: expr('Peter')},
                {x: expr('Philip'), y: expr('Zara')},
                {x: expr('Philip'), y: expr('Beatrice')},
                {x: expr('Philip'), y: expr('Eugenie')}]
examples_neg = [{x: expr('Anne'), y: expr('Eugenie')},
                {x: expr('Beatrice'), y: expr('Eugenie')},
                {x: expr('Elizabeth'), y: expr('Andrew')},
                {x: expr('Elizabeth'), y: expr('Anne')},
                {x: expr('Elizabeth'), y: expr('Mark')},
                {x: expr('Elizabeth'), y: expr('Sarah')},
                {x: expr('Philip'), y: expr('Anne')},
                {x: expr('Philip'), y: expr('Andrew')},
                {x: expr('Anne'), y: expr('Peter')},
                {x: expr('Anne'), y: expr('Zara')},
                {x: expr('Mark'), y: expr('Peter')},
                {x: expr('Mark'), y: expr('Zara')},
                {x: expr('Andrew'), y: expr('Beatrice')},
                {x: expr('Andrew'), y: expr('Eugenie')},
                {x: expr('Sarah'), y: expr('Beatrice')},
                {x: expr('Mark'), y: expr('Elizabeth')},
                {x: expr('Beatrice'), y: expr('Philip')}, 
                {x: expr('Peter'), y: expr('Andrew')}, 
                {x: expr('Zara'), y: expr('Mark')},
                {x: expr('Peter'), y: expr('Anne')},
                {x: expr('Zara'), y: expr('Eugenie')},     ]

clauses = small_family.foil([examples_pos, examples_neg], target)

print(clauses)

# %% [markdown]
# Indeed the algorithm returned the rule: 
# <br>$Grandparent(x,y) \Leftrightarrow \exists \: v \: \: Parent(x,v) \land Parent(v,y)$

# %% [markdown]
# ### Example Network
#
# Suppose that we have the following directed graph and we want to find a rule that describes the reachability between two nodes (Reach(x,y)). <br>
# Such a rule could be recursive, since y can be reached from x if and only if there is a sequence of adjacent nodes from x to y: 
#
# $$ Reach(x,y) \Leftrightarrow \begin{cases} 
#                 Conn(x,y), \: \text{(if there is a directed edge from x to y)} \\
#                 \lor \quad \exists \: z \quad Reach(x,z) \land Reach(z,y) \end{cases}$$
#

# %%
"""
A              H
|\            /|
| \          / |
v  v        v  v
B  D-->E-->G-->I
|  /   |
| /    |
vv     v
C      F
"""
small_network = FOILContainer([expr("Conn(A, B)"),
                               expr("Conn(A ,D)"),
                               expr("Conn(B, C)"),
                               expr("Conn(D, C)"),
                               expr("Conn(D, E)"),
                               expr("Conn(E ,F)"),
                               expr("Conn(E, G)"),
                               expr("Conn(G, I)"),
                               expr("Conn(H, G)"),
                               expr("Conn(H, I)")])


# %%
target = expr('Reach(x, y)')
examples_pos = [{x: A, y: B},
                {x: A, y: C},
                {x: A, y: D},
                {x: A, y: E},
                {x: A, y: F},
                {x: A, y: G},
                {x: A, y: I},
                {x: B, y: C},
                {x: D, y: C},
                {x: D, y: E},
                {x: D, y: F},
                {x: D, y: G},
                {x: D, y: I},
                {x: E, y: F},
                {x: E, y: G},
                {x: E, y: I},
                {x: G, y: I},
                {x: H, y: G},
                {x: H, y: I}]
nodes = {A, B, C, D, E, F, G, H, I}
examples_neg = [example for example in [{x: a, y: b} for a in nodes for b in nodes]
                    if example not in examples_pos]
clauses = small_network.foil([examples_pos, examples_neg], target)

print(clauses)

# %% [markdown]
# The algorithm produced something close to the recursive rule: 
#  $$ Reach(x,y) \Leftrightarrow [Conn(x,y)] \: \lor \: [\exists \: z \: \: Reach(x,z) \, \land  \, Reach(z,y)]$$
#  
# This happened because the size of the example is small. 
