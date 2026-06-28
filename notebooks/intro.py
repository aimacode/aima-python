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
# # An Introduction To `aima-python`  
#   
# The [aima-python](https://github.com/aimacode/aima-python) repository implements, in Python code, the algorithms in the textbook *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu)*. A typical module in the repository has the code for a single chapter in the book, but some modules combine several chapters. See [the index](https://github.com/aimacode/aima-python#index-of-code) if you can't find the algorithm you want. The code in this repository attempts to mirror the pseudocode in the textbook as closely as possible and to stress readability foremost; if you are looking for high-performance code with advanced features, there are other repositories for you. For each module, there are three/four files, for example:
#
# - [**`nlp.py`**](https://github.com/aimacode/aima-python/blob/master/nlp.py): Source code with data types and algorithms for natural language processing; functions have docstrings explaining their use.
# - [**`nlp.ipynb`**](https://github.com/aimacode/aima-python/blob/master/nlp.ipynb): A notebook like this one; gives more detailed examples and explanations of use.
# - [**`nlp_apps.ipynb`**](https://github.com/aimacode/aima-python/blob/master/nlp_apps.ipynb): A Jupyter notebook that gives example applications of the code.
# - [**`tests/test_nlp.py`**](https://github.com/aimacode/aima-python/blob/master/tests/test_nlp.py): Test cases, used to verify the code is correct, and also useful to see examples of use.
#
# There is also an [aima-java](https://github.com/aimacode/aima-java) repository, if you prefer Java.
#   
# ## What version of Python?
#   
# The code is tested in Python [3.4](https://www.python.org/download/releases/3.4.3/) and [3.5](https://www.python.org/downloads/release/python-351/). If you try a different version of Python 3 and find a problem, please report it as an [Issue](https://github.com/aimacode/aima-python/issues).
#   
# We recommend the [Anaconda](https://www.anaconda.com/download/) distribution of Python 3.5. It comes with additional tools like the powerful IPython interpreter, the Jupyter Notebook and many helpful packages for scientific computing. After installing Anaconda, you will be good to go to run all the code and all the IPython notebooks. 
#
# ## IPython notebooks  
#   
# The IPython notebooks in this repository explain how to use the modules, and give examples of usage. 
# You can use them in three ways: 
#
# 1. View static HTML pages. (Just browse to the [repository](https://github.com/aimacode/aima-python) and click on a `.ipynb` file link.)
# 2. Run, modify, and re-run code, live. (Download the repository (by [zip file](https://github.com/aimacode/aima-python/archive/master.zip) or by `git` commands), start a Jupyter notebook server with the shell command "`jupyter notebook`" (issued from the directory where the files are), and click on the notebook you want to interact with.)
# 3. Binder - Click on the binder badge on the [repository](https://github.com/aimacode/aima-python) main page to open the notebooks in an executable environment, online. This method does not require any extra installation. The code can be executed and modified from the browser itself. Note that this is an unstable option; there is a chance the notebooks will never load.
#
#   
# You can [read about notebooks](https://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/) and then [get started](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Running%20Code.ipynb).

# %% [markdown]
# # Helpful Tips
#
# Most of these notebooks start by importing all the symbols in a module:

# %%
from aima.logic import *

# %% [markdown]
# From there, the notebook alternates explanations with examples of use. You can run the examples as they are, and you can modify the code cells (or add new cells) and run your own examples. If you have some really good examples to add, you can make a github pull request.
#
# If you want to see the source code of a function, you can open a browser or editor and see it in another window, or from within the notebook you can use the IPython magic function `%psource` (for "print source") or the function `psource` from `notebook.py`. Also, if the algorithm has pseudocode available, you can read it by calling the `pseudocode` function with the name of the algorithm passed as a parameter.

# %%
# %psource WalkSAT

# %%
from aima.notebook_utils import psource, pseudocode

psource(WalkSAT)
pseudocode("WalkSAT")

# %% [markdown]
# Or see an abbreviated description of an object with a trailing question mark:

# %%
# WalkSAT?

# %% [markdown]
# # Authors
#
# This notebook is written by [Chirag Vertak](https://github.com/chiragvartak) and [Peter Norvig](https://github.com/norvig).
