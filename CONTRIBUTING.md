How to Contribute to aima-python
==========================

Thanks for considering contributing to `aima-python`! Here is some of the work that needs to be done:

## Port to Python 3; Pythonic Idioms; py.test

- Check for common problems in [porting to Python 3](http://python3porting.com/problems.html), such as: `print` is now a function; `range` and `map` and other functions no longer produce `list`s; objects of different types can no longer be compared with `<`; strings are now Unicode; it would be nice to move `%` string formating to `.format`; there is a new `next` function for generators; integer division now returns a float; we can now use set literals.
- Replace old Lisp-based idioms with proper Python idioms. For example, we have many functions that were taken directly from Common Lisp, such as the `every` function: `every(callable, items)` returns true if every element of `items` is callable. This is good Lisp style, but good Python style would be to use `all` and a generator expression: `all(callable(f) for f in items)`. Eventually, fix all calls to these legacy Lisp functions and then remove the functions.
- Add more tests in `_test.py` files. Strive for terseness; it is ok to group multiple asserts into one `def test_something():` function. Move most tests to `_test.py`, but it is fine to have a single `doctest` example in the docstring of a function in the `.py` file, if the purpose of the doctest is to explain how to use the function, rather than test the implementation.

## New and Improved Algorithms

- Implement functions that were in the third edition of the book but were not yet implemented in the code. Check the [list of pseudocode algorithms (pdf)](https://github.com/aimacode/pseudocode/blob/master/aima3e-algorithms.pdf) to see what's missing.
- As we finish chapters for the new fourth edition, we will share the new pseudocode in the [`aima-pseudocode`](https://github.com/aimacode/aima-pseudocode) repository, and describe what changes are necessary.
We hope to have a `algorithm-name.md` file for each algorithm, eventually; it would be great if contributors could add some for the existing algorithms.
- Give examples of how to use the code in the `.ipynb` file.

We still support a legacy branch, `aima3python2` (for the third edition of the textbook and for Python 2 code). 

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

- Follow the style guidelines described above.

Running the Test-Suite
=====================

The minimal requirement for running the testsuite is ``py.test``.  You can
install it with::

    pip install pytest

Clone this repository::

    git clone https://github.com/aimacode/aima-python.git

Fetch the aima-data submodule::

    cd aima-python
    git submodule init
    git submodule update

Then you can run the testsuite with::

    py.test

# Choice of Programming Languages

Are we right to concentrate on Java and Python versions of the code? I think so; both languages are popular; Java is
fast enough for our purposes, and has reasonable type declarations (but can be verbose); Python is popular and has a very direct mapping to the pseudocode in the book (but lacks type declarations and can be slow). The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top five most popular languages are:

        Java, C, C++, C#, Python

So it might be reasonable to also support C++/C# at some point in the future. It might also be reasonable to support a language that combines the terse readability of Python with the type safety and speed of Java; perhaps Go or Julia. And finally, Javascript is the language of the browser; it would be nice to have code that runs in the browser, in Javascript or a variant such as Typescript.

There is also a `aima-lisp` project; in 1995 when we wrote the first edition of the book, Lisp was the right choice, but today it is less popular.

What languages are instructors recommending for their AI class? To get an approximate idea, I gave the query <tt>[norvig russell "Modern Approach"](https://www.google.com/webhp#q=russell%20norvig%20%22modern%20approach%22%20java)</tt> along with the names of various languages and looked at the estimated counts of results on
various dates. However, I don't have much confidence in these figures...

|Language  |2004  |2005  |2007  |2010  |2016  |
|--------  |----: |----: |----: |----: |----: |
|[none](http://www.google.com/search?q=norvig+russell+%22Modern+Approach%22)|8,080|20,100|75,200|150,000|132,000|
|[java](http://www.google.com/search?q=java+norvig+russell+%22Modern+Approach%22)|1,990|4,930|44,200|37,000|50,000|
|[c++](http://www.google.com/search?q=c%2B%2B+norvig+russell+%22Modern+Approach%22)|875|1,820|35,300|105,000|35,000|
|[lisp](http://www.google.com/search?q=lisp+norvig+russell+%22Modern+Approach%22)|844|974|30,100|19,000|14,000|
|[prolog](http://www.google.com/search?q=prolog+norvig+russell+%22Modern+Approach%22)|789|2,010|23,200|17,000|16,000|
|[python](http://www.google.com/search?q=python+norvig+russell+%22Modern+Approach%22)|785|1,240|18,400|11,000|12,000|
