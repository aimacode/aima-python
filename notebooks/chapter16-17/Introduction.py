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

# %% [markdown]
# # Making Complex Decisions
# ---
#
# This Jupyter notebook acts as supporting material for topics covered in **Chapter 16 Making Complex Decisions** of the book *Artificial Intelligence: A Modern Approach*. We make use of the implementations in the mdp.py module. This notebook also includes a summary of the main topics as a review. The main content of this chapter is about Markov models and algorithms to solve it.

# %% [markdown]
# ## CONTENTS
#
# * Introduction
# * MDPs: the definition of Markov Decision Problems.
# * Algorithms for MDPs: Algorithms to solve MDPs: value iteration and its visualization, and policy iteration.
# * Sequential Decision Problems: demonstration case of the first example in chapter 16 of the book.
# * POMDPs: Definition of partially observed MDPs and its visualization.

# %% [markdown]
# ## OVERVIEW
#
# MDPs are meant to be a straightforward description of the real-world learning problem from interaction to achieve a goal. An agent and the environment interact continually. The agent selects actions and the environment responds to these actions and feeds new situations back to the agent.
#
# To use the implemented modules of Markov models, you need to import the packages by executing the following code in each notebook:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.mdp import *
from aima.notebook_utils import psource, pseudocode, plot_pomdp_utility

# %%
