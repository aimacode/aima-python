"""Run all doctests from modules on the command line.  Use -v for verbose.

Example usages:

    python doctests.py *.py
    python doctests.py -v *.py

You can add more module-level tests with
    __doc__ += "..."
You can add stochastic tests with
    __doc__ += random_tests("...")
"""

if __name__ == "__main__":
    import sys, glob, doctest
    args = [arg for arg in sys.argv[1:] if arg != '-v']
    if not args: args = ['*.py']
    modules = [__import__(name.replace('.py',''))
               for arg in args for name in glob.glob(arg)]
    for module in modules:
        doctest.testmod(module, report=1, optionflags=doctest.REPORT_UDIFF)
    summary = doctest.master.summarize() if modules else (0, 0)
    print '%d failed out of %d' % summary
