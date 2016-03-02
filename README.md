# aima-python: Structure of the Project

Python code for the book *Artificial Intelligence: A Modern Approach.* 
When complete, this project will cover all the major topics in the book, for each topic, such as `logic`, we will have the following [Python 3.5](https://www.python.org/downloads/release/python-350/) files:

- `logic.py`: Implementations of all the pseudocode algorithms in the book. 
- `logic_test.py`: A lightweight test suite, using `assert` statements, designed for use with `py.test`.
- `logic.ipynb`: A Jupyter notebook, with examples of usage. Does a `from logic import *` to get the code.

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

# Language Choice

Are we right to concentrate on Java and Python versions of the code? I think so; both languages are popular; Java is
fast enough for our purposes, and has reasonable type declarations (but can be verbose); Python is popular and has a very direct mapping to the pseudocode in the book (ut lacks type declarations and can be solw). The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top five most popular languages are:

        Java, C, C++, C#, Python
        
So it might be reasonable to also support C++/C# at some point in the future. It might also be reasonable to support a language that combines the terse readability of Python with the type safety and speed of Java; perhaps Go or Julia. And finally, Javascript is the language of the browser; it would be nice to have code that runs in the browser, in Javascript or a variant such as Typescript.

What languages are instructors recommending for their AI class?  
To get an approximate idea, I gave the query <tt>norvig russell "Modern Approach"</tt> along with
the names of various languages and looked at the estimated counts of results on
various dates. However, I don't have much confidence in these figures...

<p>
<table border=1>
<tr><th>Language<th>2004<th>2005<th>2007<th>2010<th>2016
<tr><td> <a href="http://www.google.com/search?q=norvig+russell+%22Modern+Approach%22"> <i>none</i></a><td align=right> 8,080<td align=right>20,100<td align=right>75,200<td align=right>150,000<td align=right>132,000
<tr><td> <a href="http://www.google.com/search?q=java+norvig+russell+%22Modern+Approach%22">java   </a><td align=right> 1,990<td align=right>4,930<td align=right>44,200<td align=right>37,000<td align=right>50,000
<tr><td> <a href="http://www.google.com/search?q=c%2B%2B+norvig+russell+%22Modern+Approach%22">c++    </a><td align=right>  875<td align=right>1,820<td align=right>35,300<td align=right>105,000<td align=right>35,000
<tr><td> <a href="http://www.google.com/search?q=lisp+norvig+russell+%22Modern+Approach%22">lisp   </a><td align=right>  844<td align=right>974<td align=right>30,100<td align=right>19,000<td align=right>14,000
<tr><td> <a href="http://www.google.com/search?q=prolog+norvig+russell+%22Modern+Approach%22">prolog </a><td align=right>  789<td align=right>2,010<td align=right>23,200<td align=right>17,000<td align=right>16,000
<tr><td> <a href="http://www.google.com/search?q=python+norvig+russell+%22Modern+Approach%22">python </a><td align=right>  785<td align=right>1,240<td align=right>18,400<td align=right>11,000<td align=right>12,000

</table>

