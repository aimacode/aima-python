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
# # Bootstrap
#
# Shared setup for the notebooks: put the repository root (the directory containing the `aima` package) on `sys.path` so `from aima import ...` works regardless of where the notebook is launched. Each notebook runs this with a single `%run bootstrap.ipynb` cell.

# %%
import os, sys
# walk up from the current directory to the repo root (the one holding `aima/`)
_root = os.path.abspath(os.getcwd())
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, 'aima')):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
