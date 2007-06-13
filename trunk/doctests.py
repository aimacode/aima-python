"""Run all doctests from modules on the command line.  For each
module, if there is a "module.txt" file, run that too.  However,
if the module.txt file contains the comment "# demo",
then the remainder of the file has its ">>>" lines executed,
but not run through doctest.  The idea is that you can use this
to demo statements that return random or otherwise variable results.

Example usage:

    python doctests.py *.py
"""

import doctest, re

def run_tests(modules, verbose=None):
    "Run tests for a list of modules; then summarize results."
    for module in modules:
        tests, demos = split_extra_tests(module.__name__ + ".txt")
        if tests:
            if '__doc__' not in dir(module):
                module.__doc__ = ''
            module.__doc__ += '\n' + tests + '\n'
        doctest.testmod(module, report=0, verbose=verbose)
        if demos:
            for stmt in re.findall(">>> (.*)", demos):
                exec stmt in module.__dict__
    doctest.master.summarize()


def split_extra_tests(filename):
    """Take a filename and, if it exists, return a 2-tuple of
    the parts before and after '# demo'."""
    try:
        contents = open(filename).read() + '# demo'
        return contents.split("# demo", 1)
    except IOError:
        return ('', '')

if __name__ == "__main__":
    import sys
    modules = [__import__(name.replace('.py',''))
               for name in sys.argv if name != "-v"]
    run_tests(modules, ("-v" in sys.argv))
