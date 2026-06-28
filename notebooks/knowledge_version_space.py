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

from aima.notebook_utils import pseudocode, psource

# %% [markdown]
# ## CONTENTS
#
# * Overview
# * Version-Space Learning

# %% [markdown]
# ## OVERVIEW
#
# Like the [learning module](https://github.com/aimacode/aima-python/blob/master/learning.ipynb), this chapter focuses on methods for generating a model/hypothesis for a domain. Unlike though the learning chapter, here we use prior knowledge to help us learn from new experiences and find a proper hypothesis.
#
# ### First-Order Logic
#
# Usually knowledge in this field is represented as **first-order logic**, a type of logic that uses variables and quantifiers in logical sentences. Hypotheses are represented by logical sentences with variables, while examples are logical sentences with set values instead of variables. The goal is to assign a value to a special first-order logic predicate, called **goal predicate**, for new examples given a hypothesis. We learn this hypothesis by infering knowledge from some given examples.
#
# ### Representation
#
# In this module, we use dictionaries to represent examples, with keys the attribute names and values the corresponding example values. Examples also have an extra boolean field, 'GOAL', for the goal predicate. A hypothesis is represented as a list of dictionaries. Each dictionary in that list represents a disjunction. Inside these dictionaries/disjunctions we have conjunctions.
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
# ## VERSION-SPACE LEARNING
#
# ### Overview
#
# **Version-Space Learning** is a general method of learning in logic based domains. We generate the set of all the possible hypotheses in the domain and then we iteratively remove hypotheses inconsistent with the examples. The set of remaining hypotheses is called **version space**. Because hypotheses are being removed until we end up with a set of hypotheses consistent with all the examples, the algorithm is sometimes called **candidate elimination** algorithm.
#
# After we update the set on an example, all the hypotheses in the set are consistent with that example. So, when all the examples have been parsed, all the remaining hypotheses in the set are consistent with all the examples. That means we can pick hypotheses at random and we will always get a valid hypothesis.

# %% [markdown]
# ### Pseudocode

# %%
pseudocode('Version-Space-Learning')

# %% [markdown]
# ### Implementation
#
# The set of hypotheses is represented by a list and each hypothesis is represented by a list of dictionaries, each dictionary a disjunction. For each example in the given examples we update the version space with the function `version_space_update`. In the end, we return the version-space.
#
# Before we can start updating the version space, we need to generate it. We do that with the `all_hypotheses` function, which builds a list of all the possible hypotheses (including hypotheses with disjunctions). The function works like this: first it finds the possible values for each attribute (using `values_table`), then it builds all the attribute combinations (and adds them to the hypotheses set) and finally it builds the combinations of all the disjunctions (which in this case are the hypotheses build by the attribute combinations).
#
# You can read the code for all the functions by running the cells below:

# %%
psource(version_space_learning, version_space_update)

# %%
psource(all_hypotheses, values_table)

# %%
psource(build_attr_combinations, build_h_combinations)

# %% [markdown]
# ### Example
#
# Since the set of all possible hypotheses is enormous and would take a long time to generate, we will come up with another, even smaller domain. We will try and predict whether we will have a party or not given the availability of pizza and soda. Let's do it:

# %%
party = [
    {'Pizza': 'Yes', 'Soda': 'No', 'GOAL': True},
    {'Pizza': 'Yes', 'Soda': 'Yes', 'GOAL': True},
    {'Pizza': 'No', 'Soda': 'No', 'GOAL': False}
]

# %% [markdown]
# Even though it is obvious that no-pizza no-party, we will run the algorithm and see what other hypotheses are valid.

# %%
V = version_space_learning(party)
for e in party:
    guess = False
    for h in V:
        if guess_value(e, h):
            guess = True
            break

    print(guess)

# %% [markdown]
# The results are correct for the given examples. Let's take a look at the version space:

# %%
print(len(V))

print(V[5])
print(V[10])

print([{'Pizza': 'Yes'}] in V)

# %% [markdown]
# There are almost 1000 hypotheses in the set. You can see that even with just two attributes the version space in very large.
#
# Our initial prediction is indeed in the set of hypotheses. Also, the two other random hypotheses we got are consistent with the examples (since they both include the "Pizza is available" disjunction).

# %% [markdown]
# ## Minimal Consistent Determination

# %% [markdown]
# This algorithm is based on a straightforward attempt to find the simplest determination consistent with the observations. A determinaton P > Q says that if any examples match on P, then they must also match on Q. A determination is therefore consistent with a set of examples if every pair that matches on the predicates on the left-hand side also matches on the goal predicate.

# %% [markdown]
# ### Pseudocode

# %% [markdown]
# Lets look at the pseudocode for this algorithm

# %%
pseudocode('Minimal-Consistent-Det')

# %% [markdown]
# You can read the code for the above algorithm by running the cells below:

# %%
psource(minimal_consistent_det)

# %%
psource(consistent_det)

# %% [markdown]
# ### Example:

# %% [markdown]
# We already know that no-pizza-no-party but we will still check it through the `minimal_consistent_det` algorithm.

# %%
print(minimal_consistent_det(party, {'Pizza', 'Soda'}))

# %% [markdown]
# We can also check it on some other example. Let's consider the following example :

# %%
conductance = [
    {'Sample': 'S1', 'Mass': 12, 'Temp': 26, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.59},
    {'Sample': 'S1', 'Mass': 12, 'Temp': 100, 'Material': 'Cu', 'Size': 3, 'GOAL': 0.57},
    {'Sample': 'S2', 'Mass': 24, 'Temp': 26, 'Material': 'Cu', 'Size': 6, 'GOAL': 0.59},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 26, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.05},
    {'Sample': 'S3', 'Mass': 12, 'Temp': 100, 'Material': 'Pb', 'Size': 2, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S4', 'Mass': 18, 'Temp': 100, 'Material': 'Pb', 'Size': 3, 'GOAL': 0.04},
    {'Sample': 'S5', 'Mass': 24, 'Temp': 100, 'Material': 'Pb', 'Size': 4, 'GOAL': 0.04},
    {'Sample': 'S6', 'Mass': 36, 'Temp': 26, 'Material': 'Pb', 'Size': 6, 'GOAL': 0.05},
]



# %% [markdown]
# Now, we check the `minimal_consistent_det` algorithm on the above example:

# %%
print(minimal_consistent_det(conductance, {'Mass', 'Temp', 'Material', 'Size'}))

# %%
print(minimal_consistent_det(conductance, {'Mass', 'Temp', 'Size'}))

