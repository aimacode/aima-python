# aima-python

Python 3 code for the book *Artificial Intelligence: A Modern Approach.*

Eventually, this repository should have code for everything in the book.

# Style Guide

We default to [Pep 8](https://www.python.org/dev/peps/pep-0008), but with a few exceptions:

- I'm not too worried about an occasional line longer than 79 characters. 
- You don't need two spaces after a sentence-ending period.
- Strunk and White is [not a good guide for English](http://chronicle.com/article/50-Years-of-Stupid-Grammar/25497).
- I prefer more concise docstrings; I don't follow [Pep 257](https://www.python.org/dev/peps/pep-0257/).
- Not all constants have to be UPPERCASE.
- [Pep 484](https://www.python.org/dev/peps/pep-0484/) type annotations are allowed but not required. If your
  parameter name is already suggestive of the name of a type, you don't need an annotation, e.g.:

        def retry(url: Url) -> None: # This 'Url' annotation should be avoided; but '-> None' is useful

# Language Popularity

Are we right to concentrate on Java and Python versions of the code?
What languages do students already know? The [TIOBE Index](http://www.tiobe.com/tiobe_index) says the top five are:

        Java, C, C++, C#, Python
        
What languages are instructors recommending for their AI class?  
To get an approximate
idea, I gave the query <tt>norvig russell "Modern Approach"</tt> along with
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

