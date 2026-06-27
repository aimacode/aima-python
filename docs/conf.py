"""Sphinx configuration for the aima-python API documentation.

The docs are built from the in-source docstrings via autodoc; heavy optional
dependencies are mocked so the build needs only Sphinx plus a few light
packages (see docs/requirements.txt), which keeps Read the Docs builds fast.
"""

import os
import sys

# make the library importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "aima-python"
copyright = "aima-python contributors"
author = "aima-python contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# heavy / optional third-party deps that need not be importable to build the docs
autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "cv2",
    "qpsolvers",
    "cvxopt",
    "cvxpy",
    "ipywidgets",
    "ipythonblocks",
    "IPython",
    "PIL",
    "qpsolvers",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
