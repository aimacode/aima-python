import pytest
from aimaPy.utils import *

eps = 0.001


def test_struct_initialization():
    s = Struct(a=1, b=2)
    assert s.a == 1
    assert s.b == 2


def test_struct_assignment():
    s = Struct(a=1)
    s.a = 3
    assert s.a == 3


def test_update_dict():
    assert update({'a': 1}, a=10, b=20) == {'a': 10, 'b': 20}
    assert update({}, a=5) == {'a': 5}


def test_update_struct():
    assert update(Struct(a=1), a=30, b=20).__cmp__(Struct(a=30, b=20))
    assert update(Struct(), a=10).__cmp__(Struct(a=10))


def test_removeall_list():
    assert removeall(4, []) == []
    assert removeall(4, [1, 2, 3, 4]) == [1, 2, 3]
    assert removeall(4, [4, 1, 4, 2, 3, 4, 4]) == [1, 2, 3]


def test_removeall_string():
    assert removeall('s', '') == ''
    assert removeall(
        's', 'This is a test. Was a test.') == 'Thi i a tet. Wa a tet.'


def test_unique():
    assert unique([1, 2, 3, 2, 1]) == [1, 2, 3]
    assert unique([1, 5, 6, 7, 6, 5]) == [1, 5, 6, 7]


def test_product():
    assert product([1, 2, 3, 4]) == 24
    assert product(list(range(1, 11))) == 3628800


def test_find_if():
    assert find_if(callable, [1, 2, 3]) is None
    assert find_if(callable, [3, min, max]) == min


def test_count_if():
    assert count_if(callable, [42, None, max, min]) == 2

    def is_odd(x):
        return x % 2

    assert count_if(is_odd, []) == 0
    assert count_if(is_odd, [1, 2, 3, 4, 5]) == 3


def test_every():
    assert every(callable, [min, max]) == 1
    assert every(callable, [min, 3]) == 0


def test_some():
    assert some(callable, [min, 3]) == 1
    assert some(callable, [2, 3]) == 0


def test_is_in():
    e = []
    assert is_in(e, [1, e, 3]) is True
    assert is_in(e, [1, [], 3]) is False


def test_argmin():
    assert argmin([-2, 1], lambda x: x**2) == 1


def test_argmin_list():
    assert argmin_list(['one', 'to', 'three', 'or'], len) == ['to', 'or']


def test_argmin_gen():
    assert [i for i in argmin_gen(['one', 'to', 'three', 'or'], len)] == [
        'to', 'or']


def test_argmax():
    assert argmax([-2, 1], lambda x: x**2) == -2
    assert argmax(['one', 'to', 'three'], len) == 'three'


def test_argmax_list():
    assert argmax_list(['one', 'three', 'seven'], lambda x: len(x)) == [
        'three', 'seven']


def test_argmax_gen():
    assert argmax_list(['one', 'three', 'seven'], len) == ['three', 'seven']


def test_histogram():
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1]) == [(1, 2), (2, 3), (4, 2), (5, 1), (7, 1), (9, 1)]
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1], 0, lambda x: x*x) == [(1, 2), (4, 3), (16, 2), (25, 1), (49, 1), (81, 1)]
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1], 1) == [(2, 3), (4, 2), (1, 2), (9, 1), (7, 1), (5, 1)]


def test_dotproduct():
    assert dotproduct([1, 2, 3], [1000, 100, 10]) == 1230


def test_vector_add():
    assert vector_add((0, 1), (8, 9)) == (8, 10)


def test_scalar_vector_product():
    assert scalar_vector_product(2, [1, 2, 3]) == [2, 4, 6]


def test_num_or_str():
    assert num_or_str('42') == 42
    assert num_or_str(' 42x ') == '42x'


def test_normalize():
    assert normalize([1, 2, 1]) == [0.25, 0.5, 0.25]


def test_clip():
    assert [clip(x, 0, 1) for x in [-1, 0.5, 10]] == [0, 0.5, 1]


def test_caller():
    assert caller(0) == 'caller'

    def f():
        return caller()
    assert f() == 'f'


def test_sigmoid():
    assert 0.5-eps <= sigmoid(0) <= 0.5+eps
    assert 0.731-eps <= sigmoid(1) <= 0.731+eps
    assert 0.269-eps <= sigmoid(-1) <= 0.269+eps


def test_sign():
    assert sign(1) == 1
    assert sign(0) == 1
    assert sign(-1) == 0


if __name__ == '__main__':
    pytest.main()
