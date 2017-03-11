import pytest
import os
import random

from text import *  # noqa
from utils import isclose, DataFile


def test_text_models():
    flatland = DataFile("EN-text/flatland.txt").read()
    wordseq = words(flatland)
    P1 = UnigramTextModel(wordseq)
    P2 = NgramTextModel(2, wordseq)
    P3 = NgramTextModel(3, wordseq)

    # The most frequent entries in each model
    assert P1.top(10) == [(2081, 'the'), (1479, 'of'), (1021, 'and'),
                          (1008, 'to'), (850, 'a'), (722, 'i'), (640, 'in'),
                          (478, 'that'), (399, 'is'), (348, 'you')]

    assert P2.top(10) == [(368, ('of', 'the')), (152, ('to', 'the')),
                          (152, ('in', 'the')), (86, ('of', 'a')),
                          (80, ('it', 'is')),
                          (71, ('by', 'the')), (68, ('for', 'the')),
                          (68, ('and', 'the')), (62, ('on', 'the')),
                          (60, ('to', 'be'))]

    assert P3.top(10) == [(30, ('a', 'straight', 'line')),
                          (19, ('of', 'three', 'dimensions')),
                          (16, ('the', 'sense', 'of')),
                          (13, ('by', 'the', 'sense')),
                          (13, ('as', 'well', 'as')),
                          (12, ('of', 'the', 'circles')),
                          (12, ('of', 'sight', 'recognition')),
                          (11, ('the', 'number', 'of')),
                          (11, ('that', 'i', 'had')), (11, ('so', 'as', 'to'))]

    assert isclose(P1['the'], 0.0611, rel_tol=0.001)

    assert isclose(P2['of', 'the'], 0.0108, rel_tol=0.01)

    assert isclose(P3['', '', 'but'], 0.0, rel_tol=0.001)
    assert isclose(P3['', '', 'but'], 0.0, rel_tol=0.001)
    assert isclose(P3['so', 'as', 'to'], 0.000323, rel_tol=0.001)

    assert P2.cond_prob.get(('went',)) is None

    assert P3.cond_prob['in', 'order'].dictionary == {'to': 6}


def test_viterbi_segmentation():
    flatland = DataFile("EN-text/flatland.txt").read()
    wordseq = words(flatland)
    P = UnigramTextModel(wordseq)
    text = "itiseasytoreadwordswithoutspaces"

    s, p = viterbi_segment(text,P)
    assert s == [
        'it', 'is', 'easy', 'to', 'read', 'words', 'without', 'spaces']


def test_shift_encoding():
    code = shift_encode("This is a secret message.", 17)

    assert code == 'Kyzj zj r jvtivk dvjjrxv.'


def test_shift_decoding():
    flatland = DataFile("EN-text/flatland.txt").read()
    ring = ShiftDecoder(flatland)
    msg = ring.decode('Kyzj zj r jvtivk dvjjrxv.')

    assert msg == 'This is a secret message.'


def test_rot13_encoding():
    code = rot13('Hello, world!')

    assert code == 'Uryyb, jbeyq!'


def test_rot13_decoding():
    flatland = DataFile("EN-text/flatland.txt").read()
    ring = ShiftDecoder(flatland)
    msg = ring.decode(rot13('Hello, world!'))

    assert msg == 'Hello, world!'


def test_counting_probability_distribution():
    D = CountingProbDist()

    for i in range(10000):
        D.add(random.choice('123456'))

    ps = [D[n] for n in '123456']

    assert 1 / 7 <= min(ps) <= max(ps) <= 1 / 5


def test_ir_system():
    from collections import namedtuple
    Results = namedtuple('IRResults', ['score', 'url'])

    uc = UnixConsultant()

    def verify_query(query, expected):
        assert len(expected) == len(query)

        for expected, (score, d) in zip(expected, query):
            doc = uc.documents[d]
            assert "{0:.2f}".format(
                expected.score) == "{0:.2f}".format(score * 100)
            assert os.path.basename(expected.url) == os.path.basename(doc.url)

        return True

    q1 = uc.query("how do I remove a file")
    assert verify_query(q1, [
        Results(76.83, "aima-data/MAN/rm.txt"),
        Results(67.83, "aima-data/MAN/tar.txt"),
        Results(67.79, "aima-data/MAN/cp.txt"),
        Results(66.58, "aima-data/MAN/zip.txt"),
        Results(64.58, "aima-data/MAN/gzip.txt"),
        Results(63.74, "aima-data/MAN/pine.txt"),
        Results(62.95, "aima-data/MAN/shred.txt"),
        Results(57.46, "aima-data/MAN/pico.txt"),
        Results(43.38, "aima-data/MAN/login.txt"),
        Results(41.93, "aima-data/MAN/ln.txt"),
    ])

    q2 = uc.query("how do I delete a file")
    assert verify_query(q2, [
        Results(75.47, "aima-data/MAN/diff.txt"),
        Results(69.12, "aima-data/MAN/pine.txt"),
        Results(63.56, "aima-data/MAN/tar.txt"),
        Results(60.63, "aima-data/MAN/zip.txt"),
        Results(57.46, "aima-data/MAN/pico.txt"),
        Results(51.28, "aima-data/MAN/shred.txt"),
        Results(26.72, "aima-data/MAN/tr.txt"),
    ])

    q3 = uc.query("email")
    assert verify_query(q3, [
        Results(18.39, "aima-data/MAN/pine.txt"),
        Results(12.01, "aima-data/MAN/info.txt"),
        Results(9.89, "aima-data/MAN/pico.txt"),
        Results(8.73, "aima-data/MAN/grep.txt"),
        Results(8.07, "aima-data/MAN/zip.txt"),
    ])

    q4 = uc.query("word count for files")
    assert verify_query(q4, [
        Results(128.15, "aima-data/MAN/grep.txt"),
        Results(94.20, "aima-data/MAN/find.txt"),
        Results(81.71, "aima-data/MAN/du.txt"),
        Results(55.45, "aima-data/MAN/ps.txt"),
        Results(53.42, "aima-data/MAN/more.txt"),
        Results(42.00, "aima-data/MAN/dd.txt"),
        Results(12.85, "aima-data/MAN/who.txt"),
    ])

    q5 = uc.query("learn: date")
    assert verify_query(q5, [])

    q6 = uc.query("2003")
    assert verify_query(q6, [
        Results(14.58, "aima-data/MAN/pine.txt"),
        Results(11.62, "aima-data/MAN/jar.txt"),
    ])


def test_words():
    assert words("``EGAD!'' Edgar cried.") == ['egad', 'edgar', 'cried']


def test_canonicalize():
    assert canonicalize("``EGAD!'' Edgar cried.") == 'egad edgar cried'


def test_translate():
    text = 'orange apple lemon '
    func = lambda x: ('s ' + x) if x ==' ' else x

    assert translate(text, func) == 'oranges  apples  lemons  '


def test_bigrams():
    assert bigrams('this') == ['th', 'hi', 'is']
    assert bigrams(['this', 'is', 'a', 'test']) == [['this', 'is'], ['is', 'a'], ['a', 'test']]


# TODO: for .ipynb
"""

>>> P1.samples(20)
'you thought known but were insides of see in depend by us dodecahedrons just but i words are instead degrees'

>>> P2.samples(20)
'flatland well then can anything else more into the total destruction and circles teach others confine women must be added'

>>> P3.samples(20)
'flatland by edwin a abbott 1884 to the wake of a certificate from nature herself proving the equal sided triangle'
"""  # noqa

if __name__ == '__main__':
    pytest.main()
