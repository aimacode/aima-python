# Introduction #

This file gives an overview of the Python code for the algorithms in
the textbook _Artificial Intelligence: A Modern
Approach_, also known as _[AIMA](http://aima.cs.berkeley.edu)_.  The code is offered free for your use under the MIT License.
As you may know, the textbook presents algorithms in
pseudo-code format; as a supplement we provide this code.
The intent is to implement all the algorithms in the book,
but we are not done yet.

# Prerequisites #

The code is meant for Python 2.5 through 2.7.

# How to Browse the Code #

You can get some use out of the code here just by browsing, starting
at the
[root of the source tree](http://code.google.com/p/aima-python/source/browse/#svn%2Ftrunk) or by clicking on the links in the
[index on the project home page](http://code.google.com/p/aima-python).
The source code is in the **.py files; the**.txt files give examples of
how to use the code.

# How to Install the Code #

If you like what you see, install the code using either one of these methods:

  1. From a command shell on your computer, execute the `svn checkout` command given on the [source tab](http://code.google.com/p/aima-python/source) of the project.  This assumes you have previously installed the version control system [Subversion](http://subversion.tigris.org/) (svn).
  1. Download and unzip the zip file listed as a "Featured download"on the right hand side of the [project home page](http://code.google.com/p/aima-python/). This is currently (Oct 2011) long out of date; we mean to make a new .zip when the svn checkout settles down.

You'll also need to install the data files from the [aima-data](http://code.google.com/p/aima-data) project.  These are text files that are used by the tests in the aima-python project, and may be useful for yout own work.

You can put the code anywhere you want on your computer, but it should be in one
directory (you might call it _aima_ but you are free to use whatever name you want) with _aima-python_ as a subdirectory that contains all the files from this project, and _data_ as a parallel subdirectory that contains all the files from the aima-data project.

# How to Test the Code #

First, you need to install Python (version 2.5 through 2.7; parts of the code may work in other versions, but don't expect it to). Python comes preinstalled on most versions of Linux and Mac OS. Versions are also available for Windows, Solaris, and other operating systems. If your system does not have Python installed, you can [download](http://python.org/download/) and install it for free.

In the _aima-python_ directory, execute the command

> `python doctests.py -v *.py`

The "-v" is optional; it means "verbose". Various output is printed, but if all goes well there should be no instances of the word "`Failure`", nor of a long line of "**".
If you do use the "-v" option, the last line printed should be "Test passed."**

# How to Run the Code #

You're on your own -- experiment!  Create a new python file, import the modules you need,
and call the functions you want.

# Acknowledgements #

Many thanks for the bug reports, corrected code, and other support from Phil Ruggera, Peng Shao, Amit Patil, Ted Nienstedt, Jim Martin, Ben Catanzariti, and others.