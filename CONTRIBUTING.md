How to Contribute to aima-python
==========================

Thanks for considering contributing to aima-python.

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

- Ensure that the patch is python 3.5 compliant.

- Include tests if your patch is supposed to solve a bug, and explain
  clearly under which circumstances the bug happens. Make sure the test fails
  without your patch.

- Try to follow `PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`_, but you
  may ignore the line-length-limit if following it would make the code uglier.

Running the Test-Suite
=====================

The minimal requirement for running the testsuite is ``py.test``.  You can
install it with::

    pip install pytest

Clone this repository::

    git clone https://github.com/aimacode/aima-python.git

Then you can run the testsuite with::

    py.test
