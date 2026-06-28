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
# * Current-Best Learning

# %% [markdown]
# ## OVERVIEW
#
# Like the [learning module](https://github.com/aimacode/aima-python/blob/master/learning.ipynb), this chapter focuses on methods for generating a model/hypothesis for a domain; however, unlike the learning chapter, here we use prior knowledge to help us learn from new experiences and find a proper hypothesis.
#
# ### First-Order Logic
#
# Usually knowledge in this field is represented as **first-order logic**; a type of logic that uses variables and quantifiers in logical sentences. Hypotheses are represented by logical sentences with variables, while examples are logical sentences with set values instead of variables. The goal is to assign a value to a special first-order logic predicate, called **goal predicate**, for new examples given a hypothesis. We learn this hypothesis by infering knowledge from some given examples.
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
# ## CURRENT-BEST LEARNING
#
# ### Overview
#
# In **Current-Best Learning**, we start with a hypothesis and we refine it as we iterate through the examples. For each example, there are three possible outcomes: the example is consistent with the hypothesis, the example is a **false positive** (real value is false but got predicted as true) and the example is a **false negative** (real value is true but got predicted as false). Depending on the outcome we refine the hypothesis accordingly:
#
# * Consistent: We do not change the hypothesis and move on to the next example.
#
# * False Positive: We **specialize** the hypothesis, which means we add a conjunction.
#
# * False Negative: We **generalize** the hypothesis, either by removing a conjunction or a disjunction, or by adding a disjunction.
#
# When specializing or generalizing, we should make sure to not create inconsistencies with previous examples. To avoid that caveat, backtracking is needed. Thankfully, there is not just one specialization or generalization, so we have a lot to choose from. We will go through all the specializations/generalizations and we will refine our hypothesis as the first specialization/generalization consistent with all the examples seen up to that point.

# %% [markdown]
# ### Pseudocode

# %%
pseudocode('Current-Best-Learning')

# %% [markdown]
# ### Implementation
#
# As mentioned earlier, examples are dictionaries (with keys being the attribute names) and hypotheses are lists of dictionaries (each dictionary is a disjunction). Also, in the hypothesis, we denote the *NOT* operation with an exclamation mark (!).
#
# We have functions to calculate the list of all specializations/generalizations, to check if an example is consistent/false positive/false negative with a hypothesis. We also have an auxiliary function to add a disjunction (or operation) to a hypothesis, and two other functions to check consistency of all (or just the negative) examples.
#
# You can read the source by running the cell below:

# %%
psource(current_best_learning, specializations, generalizations)

# %% [markdown]
# You can view the auxiliary functions in the [knowledge module](https://github.com/aimacode/aima-python/blob/master/knowledge.py). A few notes on the functionality of some of the important methods:

# %% [markdown]
# * `specializations`: For each disjunction in the hypothesis, it adds a conjunction for values in the examples encountered so far (if the conjunction is consistent with all the examples). It returns a list of hypotheses.
#
# * `generalizations`: It adds to the list of hypotheses in three phases. First it deletes disjunctions, then it deletes conjunctions and finally it adds a disjunction.
#
# * `add_or`: Used by `generalizations` to add an *or operation* (a disjunction) to the hypothesis. Since the last example is the problematic one which wasn't consistent with the hypothesis, it will model the new disjunction to that example. It creates a disjunction for each combination of attributes in the example and returns the new hypotheses consistent with the negative examples encountered so far. We do not need to check the consistency of positive examples, since they are already consistent with at least one other disjunction in the hypotheses' set, so this new disjunction doesn't affect them. In other words, if the value of a positive example is negative under the disjunction, it doesn't matter since we know there exists a disjunction consistent with the example.

# %% [markdown]
# Since the algorithm stops searching the specializations/generalizations after the first consistent hypothesis is found, usually you will get different results each time you run the code.

# %% [markdown]
# ### Examples
#
# We will take a look at two examples. The first is a trivial one, while the second is a bit more complicated (you can also find it in the book).
#
# Earlier, we had the "animals taking umbrellas" example. Now we want to find a hypothesis to predict whether or not an animal will take an umbrella. The attributes are `Species`, `Rain` and `Coat`. The possible values are `[Cat, Dog]`, `[Yes, No]` and `[Yes, No]` respectively. Below we give seven examples (with `GOAL` we denote whether an animal will take an umbrella or not):

# %%
animals_umbrellas = [
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': True},
    {'Species': 'Cat', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'Yes', 'GOAL': True},
    {'Species': 'Dog', 'Rain': 'Yes', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Dog', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'No', 'GOAL': False},
    {'Species': 'Cat', 'Rain': 'No', 'Coat': 'Yes', 'GOAL': True}
]

# %% [markdown]
# Let our initial hypothesis be `[{'Species': 'Cat'}]`. That means every cat will be taking an umbrella. We can see that this is not true, but it doesn't matter since we will refine the hypothesis using the Current-Best algorithm. First, let's see how that initial hypothesis fares to have a point of reference.

# %%
initial_h = [{'Species': 'Cat'}]

for e in animals_umbrellas:
    print(guess_value(e, initial_h))

# %% [markdown]
# We got 5/7 correct. Not terribly bad, but we can do better. Lets now run the algorithm and see how that performs in comparison to our current result. 

# %%
h = current_best_learning(animals_umbrellas, initial_h)

for e in animals_umbrellas:
    print(guess_value(e, h))

# %% [markdown]
# We got everything right! Let's print our hypothesis:

# %%
print(h)


# %% [markdown]
# If an example meets any of the disjunctions in the list, it will be `True`, otherwise it will be `False`.
#
# Let's move on to a bigger example, the "Restaurant" example from the book. The attributes for each example are the following:
#
# * Alternative option (`Alt`)
# * Bar to hang out/wait (`Bar`)
# * Day is Friday (`Fri`)
# * Is hungry (`Hun`)
# * How much does it cost (`Price`, takes values in [$, $$, $$$])
# * How many patrons are there (`Pat`, takes values in [None, Some, Full])
# * Is raining (`Rain`)
# * Has made reservation (`Res`)
# * Type of restaurant (`Type`, takes values in [French, Thai, Burger, Italian])
# * Estimated waiting time (`Est`, takes values in [0-10, 10-30, 30-60, >60])
#
# We want to predict if someone will wait or not (Goal = WillWait). Below we show twelve examples found in the book.

# %% [markdown]
# ![restaurant](images/restaurant.png)

# %% [markdown]
# With the function `r_example` we will build the dictionary examples:

# %%
def r_example(Alt, Bar, Fri, Hun, Pat, Price, Rain, Res, Type, Est, GOAL):
    return {'Alt': Alt, 'Bar': Bar, 'Fri': Fri, 'Hun': Hun, 'Pat': Pat,
            'Price': Price, 'Rain': Rain, 'Res': Res, 'Type': Type, 'Est': Est,
            'GOAL': GOAL}


# %% [markdown]
# In code:

# %%
restaurant = [
    r_example('Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', True),
    r_example('Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', False),
    r_example('No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', True),
    r_example('Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'No', 'Thai', '10-30', True),
    r_example('Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', False),
    r_example('No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', True),
    r_example('No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', False),
    r_example('No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', True),
    r_example('No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', False),
    r_example('No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', False),
    r_example('Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', True)
]

# %% [markdown]
# Say our initial hypothesis is that there should be an alternative option and lets run the algorithm.

# %%
initial_h = [{'Alt': 'Yes'}]
h = current_best_learning(restaurant, initial_h)
for e in restaurant:
    print(guess_value(e, h))

# %% [markdown]
# The predictions are correct. Let's see the hypothesis that accomplished that:

# %%
print(h)

# %% [markdown]
# It might be quite complicated, with many disjunctions if we are unlucky, but it will always be correct, as long as a correct hypothesis exists.

# %%
