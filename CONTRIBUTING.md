How to Contribute to aima-python
==========================

Thanks for considering contributing to `aima-python`! Here is some of the work that needs to be done:

## Port to Python 3; Pythonic Idioms; py.test

- Check for common problems in [porting to Python 3](http://python3porting.com/problems.html), such as: `print` is now a function; `range` and `map` and other functions no longer produce `list`s; objects of different types can no longer be compared with `<`; strings are now Unicode; it would be nice to move `%` string formating to `.format`; there is a new `next` function for generators; integer division now returns a float; we can now use set literals.
- Replace old Lisp-based idioms with proper Python idioms. For example, we have many functions that were taken directly from Common Lisp, such as the `every` function: `every(callable, items)` returns true if every element of `items` is callable. This is good Lisp style, but good Python style would be to use `all` and a generator expression: `all(callable(f) for f in items)`. Eventually, fix all calls to these legacy Lisp functions and then remove the functions.
- Add more tests in `_test.py` files. Strive for terseness; it is ok to group multiple asserts into one `def test_something():` function. Move most tests to `_test.py`, but it is fine to have a single `doctest` example in the docstring of a function in the `.py` file, if the purpose of the doctest is to explain how to use the function, rather than test the implementation.

# Style Guide

There are a few style rules that are unique to this project:

- The first rule is that the code should correspond directly to the pseudocode in the book. When possible this will be almost one-to-one, just allowing for the syntactic differences between Python and pseudocode, and for different library functions.
- Don't make a function more complicated than the pseudocode in the book, even if the complication would add a nice feature, or give an efficiency gain. Instead, remain faithful to the pseudocode, and if you must, add a new function (not in the book) with the added feature.
- I use functional programming (functions with no side effects) in many cases, but not exclusively (sometimes classes and/or functions with side effects are used). Let the book's pseudocode be the guide.

Beyond the above rules, we use [Pep 8](https://www.python.org/dev/peps/pep-0008), with a few minor exceptions:

- I have set `--max-line-length 100`, not 79.
- You don't need two spaces after a sentence-ending period.
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings; I don't follow [Pep 257](https://www.python.org/dev/peps/pep-0257/).
- Not all constants have to be UPPERCASE.
- At some point I may add [Pep 484](https://www.python.org/dev/peps/pep-0484/) type annotations, but I think I'll hold off for now;
  I want to get more experience with them, and some people may still be in Python 3.4.


Contributing a Patch
====================

1. Submit an issue describing your proposed change to the repo in question.
1. The repo owner will respond to your issue promptly.
1. Fork the desired repo, develop and test your code changes.
1. Submit a pull request.

Reporting Issues
================

- Under which versions of Python does this happen?

- Is anybody working on this?

Patch Rules
===========

- Ensure that the patch is python 3.4 compliant.

- Include tests if your patch is supposed to solve a bug, and explain
  clearly under which circumstances the bug happens. Make sure the test fails
  without your patch.

- Follw the style guidelines described above.

Running the Test-Suite
=====================

The minimal requirement for running the testsuite is ``py.test``.  You can
install it with::

    pip install pytest

Clone this repository::

    git clone https://github.com/aimacode/aima-python.git

Then you can run the testsuite with::

    py.test
