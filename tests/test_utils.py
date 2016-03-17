import pytest
from utils import *  # noqa


def test_update_dict():
    assert update({'a': 1}, a=10, b=20) == {'a': 10, 'b': 20}
    assert update({}, a=5) == {'a': 5}


def test_removeall_list():
    assert removeall(4, []) == []
    assert removeall(4, [1, 2, 3, 4]) == [1, 2, 3]
    assert removeall(4, [4, 1, 4, 2, 3, 4, 4]) == [1, 2, 3]


def test_removeall_string():
    assert removeall('s', '') == ''
    assert removeall('s', 'This is a test. Was a test.') == 'Thi i a tet. Wa a tet.'


def test_unique():
    assert unique([1, 2, 3, 2, 1]) == [1, 2, 3]
    assert unique([1, 5, 6, 7, 6, 5]) == [1, 5, 6, 7]


def test_product():
    assert product([1, 2, 3, 4]) == 24
    assert product(list(range(1, 11))) == 3628800


def test_first():
    assert first('word') == 'w'
    assert first('') is None
    assert first('', 'empty') == 'empty'
    assert first(range(10)) == 0
    assert first(x for x in range(10) if x > 3) == 4
    assert first(x for x in range(10) if x > 100) is None


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
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1]) == [(1, 2), (2, 3),
                                                         (4, 2), (5, 1),
                                                         (7, 1), (9, 1)]
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1], 0, lambda x: x*x) == [(1, 2), (4, 3),
                                                                           (16, 2), (25, 1),
                                                                           (49, 1), (81, 1)]
    assert histogram([1, 2, 4, 2, 4, 5, 7, 9, 2, 1], 1) == [(2, 3), (4, 2),
                                                            (1, 2), (9, 1),
                                                            (7, 1), (5, 1)]


def test_dotproduct():
    assert dotproduct([1, 2, 3], [1000, 100, 10]) == 1230

def test_element_wise_product():
    assert element_wise_product([1, 2, 5], [7, 10, 0]) == [7, 20, 0]
    assert element_wise_product([1, 6, 3, 0], [9, 12, 0, 0]) == [9, 72, 0, 0]


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
    assert isclose(0.5, sigmoid(0))
    assert isclose(0.7310585786300049, sigmoid(1))
    assert isclose(0.2689414213699951, sigmoid(-1))


def test_step():
    assert step(1) == 1
    assert step(0) == 1
    assert step(-1) == 0


if __name__ == '__main__':
    pytest.main()
