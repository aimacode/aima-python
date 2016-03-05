# `aima-python`: Structure of the Project

Python code for the book *Artificial Intelligence: A Modern Approach.* We're loooking for one student sponsored by Google Summer of Code (GSoC) to work on this project; if you want to be that student, make some good contributions here (by looking throush the i"Issues" and resolving some), and submit an application.

When complete, this project will cover all the major topics in the book, for each topic, such as `logic`, we will have the following [Python 3.5](https://www.python.org/downloads/release/python-350/) files in the main branch:

- `logic.py`: Implementations of all the pseudocode algorithms in the book.
- `logic_test.py`: A lightweight test suite, using `assert` statements, designed for use with `py.test`.
- `logic.ipynb`: A Jupyter notebook, with examples of usage. Does a `from logic import *` to get the code.

Until we get there, we will support a legacy branch, `aima3python2` (for the third edition of the textbook and for Python 2 code). To prepare code for the new master branch, the following two steps should be taken:

## Port to Python 3; Pythonic Idioms; py.test

- Check for common problems in [porting to Python 3](http://python3porting.com/problems.html), such as: `print` is now a function; `range` and `map` and other functions no longer produce `list`s; objects of different types can no longer be compared with `<`; strings are now Unicode; it would be nice to move `%` string formating to `.format`; there is a new `next` function for generators; integer division now returns a float; we can now use set literals.
- Replace poor idioms with proper Python. For example, we have many functions that were taken directly from Common Lisp, such as the `every` function: `every(callable, items)` returns true if every element of `items` is callable. This is good Lisp style, but good Python style would be to use `all` and a generator expression: `all(callable(f) for f in items)`. Eventually, fix all calls to these legacy Lisp functions and then remove the functions.
- Create a `_test.py` file, and define functions that use `assert` to make tests. Remove any old `doctest` tests.
In other words, replace the ">>> 2 + 2 \n 4"  in a docstring with "assert 2 + 2 == 4" in `filename_test.py`.

## New and Improved Algorithms

- Implement functions that were in the third edition of the book but were not yet implemented in the code. Check th [list of pseudocode algorithms (pdf)](http://aima.cs.berkeley.edu/algorithms.pdf) to see what's missing.
- As we finish chapters for the new fourth edition, we will share the new pseudocode, and describe what changes are necessary.

- Create a `.ipynb` notebook, and give examples of how to use the code.

# Style Guide

There are a few style rules that are unique to this project:

- The first rule is that the code should correspond directly to the pseudocode in the book. When possible this will be almost one-to-one, just allowing for the syntactic differences between Python and pseudocode, and for different library functions.
- Don't make a function more complicated than the pseudocode in the book, even if the complication would add a nice feature, or give an efficiency gain. Instead, remain faithful to the pseudocode, and if you must, add a new function (not in the book) with the added feature.
- I use functional programming (functions with no side effects) in many cases, but not exclusively (sometimes classes and/or functions with side effects are used). Let the book's pseudocode be the guide.

Beyond the above rules, we use [Pep 8](https://www.python.org/dev/peps/pep-0008), with a few minor exceptions:

- I'm not too worried about an occasional line longer than 79 characters.
- You don't need two spaces after a sentence-ending period.
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings; I don't follow [Pep 257](https://www.python.org/dev/peps/pep-0257/).
- Not all constants have to be UPPERCASE.
- [Pep 484](https://www.python.org/dev/peps/pep-0484/) type annotations are allowed but not required. If your
  parameter name is already suggestive of the name of a type, such as `url` below, then i don't think the type annotation is useful.
  Return type annotations, such as `-> None` below, can be very useful.

        def retry(url: Url) -> None:

# Choice of Programming Languages

Are we right to concentrate on Java and Python versions of the code? I think so; both languages are popular; Java is
fast enough for our purposes, and has reasonable type declarations (but can be verbose); Python is popular and has a very direct mapping to the pseudocode in the book (ut lacks type declarations and can be solw). The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top five most popular languages are:

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
