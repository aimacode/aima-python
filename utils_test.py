import pytest
from utils import *

def test_struct_initialization():
    s = Struct(a=1, b=2)
    assert s.a == 1
    assert s.b == 2

def test_struct_assignment():
    s = Struct(a=1)
    s.a = 3
    assert s.a == 3

def test_removeall_list():
    assert removeall(4, []) == []
    assert removeall(4, [1,2,3,4]) == [1,2,3]

def test_removeall_string():
    assert removeall('s', '') == ''
    assert removeall('s', 'This is a test. Was a test.') == 'Thi i a tet. Wa a tet.'

def test_count_if():
    is_odd = lambda x: x % 2
    assert count_if(is_odd, []) == 0
    assert count_if(is_odd, [1, 2, 3, 4, 5]) == 3

def test_argmax():
    assert argmax([-2, 1], lambda x: x**2) == -2

def test_argmin():
    assert argmin([-2, 1], lambda x: x**2) == 1

if __name__ == '__main__':
    pytest.main()
