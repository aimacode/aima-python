How to Contribute to aima-python
==========================

Thanks for considering contributing to `aima-python`! Whether you are an aspiring [Google Summer of Code](https://summerofcode.withgoogle.com/organizations/5431334980288512/) student, or an independent contributor, here is a guide on how you can help.

First of all, you can read these write-ups from past GSoC students to get an idea about what you can do for the project. [Chipe1](https://github.com/aimacode/aima-python/issues/641) - [MrDupin](https://github.com/aimacode/aima-python/issues/632)

In general, the main ways you can contribute to the repository are the following:

1. Implement algorithms from the [list of algorithms](https://github.com/aimacode/aima-python/blob/master/README.md#index-of-algorithms).
1. Add tests for algorithms.
1. Take care of [issues](https://github.com/aimacode/aima-python/issues).
1. Write on the notebooks (`.ipynb` files).
1. Add and edit documentation (the docstrings in `.py` files).

In more detail:

## Read the Code and Start on an Issue

- First, read and understand the code to get a feel for the extent and the style.
- Look at the [issues](https://github.com/aimacode/aima-python/issues) and pick one to work on.
- One of the issues is that some algorithms are missing from the [list of algorithms](https://github.com/aimacode/aima-python/blob/master/README.md#index-of-algorithms) and that some don't have tests.

## Port to Python 3; Pythonic Idioms

- Check for common problems in [porting to Python 3](http://python3porting.com/problems.html), such as: `print` is now a function; `range` and `map` and other functions no longer produce `list`; objects of different types can no longer be compared with `<`; strings are now Unicode; it would be nice to move `%` string formatting to `.format`; there is a new `next` function for generators; integer division now returns a float; we can now use set literals.
- Replace old Lisp-based idioms with proper Python idioms. For example, we have many functions that were taken directly from Common Lisp, such as the `every` function: `every(callable, items)` returns true if every element of `items` is callable. This is good Lisp style, but good Python style would be to use `all` and a generator expression: `all(callable(f) for f in items)`. Eventually, fix all calls to these legacy Lisp functions and then remove the functions.

## New and Improved Algorithms

- Implement functions that were in the third edition of the book but were not yet implemented in the code. Check the [list of pseudocode algorithms (pdf)](https://github.com/aimacode/pseudocode/blob/master/aima3e-algorithms.pdf) to see what's missing.
- As we finish chapters for the new fourth edition, we will share the new pseudocode in the [`aima-pseudocode`](https://github.com/aimacode/aima-pseudocode) repository, and describe what changes are necessary.
We hope to have an `algorithm-name.md` file for each algorithm, eventually; it would be great if contributors could add some for the existing algorithms.

## Jupyter Notebooks

In this project we use Jupyter/IPython Notebooks to showcase the algorithms in the book. They serve as short tutorials on what the algorithms do, how they are implemented and how one can use them. To install Jupyter, you can follow the instructions [here](https://jupyter.org/install.html). These are some ways you can contribute to the notebooks:

- Proofread the notebooks for grammar mistakes, typos, or general errors.
- Move visualization and unrelated to the algorithm code from notebooks to `notebook.py` (a file used to store code for the notebooks, like visualization and other miscellaneous stuff). Make sure the notebooks still work and have their outputs showing!
- Replace the `%psource` magic notebook command with the function `psource` from `notebook.py` where needed. Examples where this is useful are a) when we want to show code for algorithm implementation and b) when we have consecutive cells with the magic keyword (in this case, if the code is large, it's best to leave the output hidden).
- Add the function `pseudocode(algorithm_name)` in algorithm sections. The function prints the pseudocode of the algorithm. You can see some example usage in [`knowledge.ipynb`](https://github.com/aimacode/aima-python/blob/master/knowledge.ipynb).
- Edit existing sections for algorithms to add more information and/or examples.
- Add visualizations for algorithms. The visualization code should go in `notebook.py` to keep things clean.
- Add new sections for algorithms not yet covered. The general format we use in the notebooks is the following: First start with an overview of the algorithm, printing the pseudocode and explaining how it works. Then, add some implementation details, including showing the code (using `psource`). Finally, add examples for the implementations, showing how the algorithms work. Don't fret with adding complex, real-world examples; the project is meant for educational purposes. You can of course choose another format if something better suits an algorithm.

Apart from the notebooks explaining how the algorithms work, we also have notebooks showcasing some indicative applications of the algorithms. These notebooks are in the `*_apps.ipynb` format. We aim to have an `apps` notebook for each module, so if you don't see one for the module you would like to contribute to, feel free to create it from scratch! In these notebooks we are looking for applications showing what the algorithms can do. The general format of these sections is this: Add a description of the problem you are trying to solve, then explain how you are going to solve it and finally provide your solution with examples. Note that any code you write should not require any external libraries apart from the ones already provided (like `matplotlib`).

# Style Guide

There are a few style rules that are unique to this project:

- The first rule is that the code should correspond directly to the pseudocode in the book. When possible this will be almost one-to-one, just allowing for the syntactic differences between Python and pseudocode, and for different library functions.
- Don't make a function more complicated than the pseudocode in the book, even if the complication would add a nice feature, or give an efficiency gain. Instead, remain faithful to the pseudocode, and if you must, add a new function (not in the book) with the added feature.
- I use functional programming (functions with no side effects) in many cases, but not exclusively (sometimes classes and/or functions with side effects are used). Let the book's pseudocode be the guide.

Beyond the above rules, we use [Pep 8](https://www.python.org/dev/peps/pep-0008), with a few minor exceptions:

- I have set `--max-line-length 100`, not 79.
- You don't need two spaces after a sentence-ending period.
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings; I don't follow [Pep 257](https://www.python.org/dev/peps/pep-0257/). In most cases,
a one-line docstring suffices. It is rarely necessary to list what each argument does; the name of the argument usually is enough.
- Not all constants have to be UPPERCASE.
- At some point I may add [Pep 484](https://www.python.org/dev/peps/pep-0484/) type annotations, but I think I'll hold off for now;
  I want to get more experience with them, and some people may still be in Python 3.4.

Reporting Issues
================

- Under which versions of Python does this happen?

- Provide an example of the issue occurring.

- Is anybody working on this?

Patch Rules
===========

- Ensure that the patch is Python 3.4 compliant.

- Include tests if your patch is supposed to solve a bug, and explain
  clearly under which circumstances the bug happens. Make sure the test fails
  without your patch.

- Follow the style guidelines described above.
- Refer the issue you have fixed.
- Explain in brief what changes you have made with affected files name.

# Choice of Programming Languages

Are we right to concentrate on Java and Python versions of the code? I think so; both languages are popular; Java is
fast enough for our purposes, and has reasonable type declarations (but can be verbose); Python is popular and has a very direct mapping to the pseudocode in the book (but lacks type declarations and can be slow). The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top seven most popular languages, in order, are:

        Java, C, C++, C#, Python, PHP, Javascript

So it might be reasonable to also support C++/C# at some point in the future. It might also be reasonable to support a language that combines the terse readability of Python with the type safety and speed of Java; perhaps Go or Julia. I see no reason to support PHP. Javascript is the language of the browser; it would be nice to have code that runs in the browser without need for any downloads; this would be in Javascript or a variant such as Typescript.

There is also a `aima-lisp` project; in 1995 when we wrote the first edition of the book, Lisp was the right choice, but today it is less popular (currently #31 on the TIOBE index).

What languages are instructors recommending for their AI class? To get an approximate idea, I gave the query <tt>[\[norvig russell "Modern Approach"\]](https://www.google.com/webhp#q=russell%20norvig%20%22modern%20approach%22%20java)</tt> along with the names of various languages and looked at the estimated counts of results on
various dates. However, I don't have much confidence in these figures...

|Language  |2004  |2005  |2007  |2010  |2016  |
|--------  |----: |----: |----: |----: |----: |
|[none](http://www.google.com/search?q=norvig+russell+%22Modern+Approach%22)|8,080|20,100|75,200|150,000|132,000|
|[java](http://www.google.com/search?q=java+norvig+russell+%22Modern+Approach%22)|1,990|4,930|44,200|37,000|50,000|
|[c++](http://www.google.com/search?q=c%2B%2B+norvig+russell+%22Modern+Approach%22)|875|1,820|35,300|105,000|35,000|
|[lisp](http://www.google.com/search?q=lisp+norvig+russell+%22Modern+Approach%22)|844|974|30,100|19,000|14,000|
|[prolog](http://www.google.com/search?q=prolog+norvig+russell+%22Modern+Approach%22)|789|2,010|23,200|17,000|16,000|
|[python](http://www.google.com/search?q=python+norvig+russell+%22Modern+Approach%22)|785|1,240|18,400|11,000|12,000|
