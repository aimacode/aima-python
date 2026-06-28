"""Sphinx configuration for the aima-python API documentation.

The docs are built from the in-source docstrings via autodoc; heavy optional
dependencies are mocked so the build needs only Sphinx plus a few light
packages (see docs/requirements.txt), which keeps Read the Docs builds fast.
"""

import os
import re
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

# show members as ``expr()`` rather than ``aima.utils.expr()`` -- cleaner headings
add_module_names = False

# napoleon: render Google/NumPy-style sections a bit more compactly
napoleon_use_rtype = False
napoleon_use_ivar = True


def _strip_doctests(app, what, name, obj, options, lines):
    """Drop doctest (``>>> ...``) example blocks from the rendered API docs.

    The examples stay in the source docstrings (useful when reading the code or
    running them as doctests); they are merely omitted from the HTML so the API
    reference reads as clean prose instead of interactive-session transcripts.
    """
    cleaned, in_block = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            if not in_block:
                # also drop a now-dangling "Example:"/"Examples:" header + blanks
                while cleaned and (not cleaned[-1].strip()
                                   or cleaned[-1].strip().rstrip(":").lower()
                                   in ("example", "examples", "for example")):
                    cleaned.pop()
                # also drop a trailing "... Example:" sentence on the prose line
                if cleaned:
                    cleaned[-1] = re.sub(
                        r"\.\s+(for example|examples?|e\.g\.)\s*:?\s*$", ".",
                        cleaned[-1], flags=re.I)
                in_block = True
            continue
        if in_block:
            if not stripped:            # a blank line ends the doctest block
                in_block = False
                cleaned.append(line)
            # otherwise: a ``...`` continuation or an expected-output line -> drop
            continue
        cleaned.append(line)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    lines[:] = cleaned


def setup(app):
    app.connect("autodoc-process-docstring", _strip_doctests)


# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
