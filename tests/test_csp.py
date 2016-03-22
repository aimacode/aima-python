import pytest
from csp import *   #noqa

def test_backtracking_search():
    assert (backtracking_search(australia) is not None) == True

if __name__ == "__main__":
    pytest.main()
