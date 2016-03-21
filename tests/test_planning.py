import pytest
from planning import *  # noqa

# flake8: noqa

romania = GraphProblem('Arad', 'Bucharest', Fig[3, 2])


def test_hierarchicalSearch():
    assert hierarchicalSearch(romania).solution() == ['Sibiu', 'Fagaras', 'Bucharest']
    
    
if __name__ == '__main__':
    pytest.main()
