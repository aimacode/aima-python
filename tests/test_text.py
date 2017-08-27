import pytest
import os
import random

from text import *
from utils import isclose, open_data



def test_text_models():
    flatland = open_data("EN-text/flatland.txt").read()
    wordseq = words(flatland)
    P1 = UnigramWordModel(wordseq)
    P2 = NgramWordModel(2, wordseq)
    P3 = NgramWordModel(3, wordseq)

    # Test top
    assert P1.top(5) == [(2081, 'the'), (1479, 'of'),
                         (1021, 'and'), (1008, 'to'),
                         (850, 'a')]

    assert P2.top(5) == [(368, ('of', 'the')), (152, ('to', 'the')),
                         (152, ('in', 'the')), (86, ('of', 'a')),
                         (80, ('it', 'is'))]

    assert P3.top(5) == [(30, ('a', 'straight', 'line')),
                         (19, ('of', 'three', 'dimensions')),
                         (16, ('the', 'sense', 'of')),
                         (13, ('by', 'the', 'sense')),
                         (13, ('as', 'well', 'as'))]

    # Test isclose
    assert isclose(P1['the'], 0.0611, rel_tol=0.001)
    assert isclose(P2['of', 'the'], 0.0108, rel_tol=0.01)
    assert isclose(P3['so', 'as', 'to'], 0.000323, rel_tol=0.001)

    # Test cond_prob.get
    assert P2.cond_prob.get(('went',)) is None
    assert P3.cond_prob['in', 'order'].dictionary == {'to': 6}

    # Test dictionary
    test_string = 'unigram'
    wordseq = words(test_string)
    P1 = UnigramWordModel(wordseq)
    assert P1.dictionary == {('unigram'): 1}

    test_string = 'bigram text'
    wordseq = words(test_string)
    P2 = NgramWordModel(2, wordseq)
    assert P2.dictionary == {('bigram', 'text'): 1}

    test_string = 'test trigram text here'
    wordseq = words(test_string)
    P3 = NgramWordModel(3, wordseq)
    assert ('test', 'trigram', 'text') in P3.dictionary
    assert ('trigram', 'text', 'here') in P3.dictionary


def test_char_models():
    test_string = 'test unigram'
    wordseq = words(test_string)
    P1 = UnigramCharModel(wordseq)

    expected_unigrams = {'n': 1, 's': 1, 'e': 1, 'i': 1, 'm': 1, 'g': 1, 'r': 1, 'a': 1, 't': 2, 'u': 1}
    assert len(P1.dictionary) == len(expected_unigrams)
    for char in test_string.replace(' ', ''):
        assert char in P1.dictionary

    test_string = 'alpha beta'
    wordseq = words(test_string)
    P1 = NgramCharModel(1, wordseq)

    assert len(P1.dictionary) == len(set(test_string))
    for char in set(test_string):
        assert tuple(char) in P1.dictionary

    test_string = 'bigram'
    wordseq = words(test_string)
    P2 = NgramCharModel(2, wordseq)

    expected_bigrams = {(' ', 'b'): 1, ('b', 'i'): 1, ('i', 'g'): 1, ('g', 'r'): 1, ('r', 'a'): 1, ('a', 'm'): 1}

    assert len(P2.dictionary) == len(expected_bigrams)
    for bigram, count in expected_bigrams.items():
        assert bigram in P2.dictionary
        assert P2.dictionary[bigram] == count

    test_string = 'bigram bigram'
    wordseq = words(test_string)
    P2 = NgramCharModel(2, wordseq)

    expected_bigrams = {(' ', 'b'): 2, ('b', 'i'): 2, ('i', 'g'): 2, ('g', 'r'): 2, ('r', 'a'): 2, ('a', 'm'): 2}

    assert len(P2.dictionary) == len(expected_bigrams)
    for bigram, count in expected_bigrams.items():
        assert bigram in P2.dictionary
        assert P2.dictionary[bigram] == count

    test_string = 'trigram'
    wordseq = words(test_string)
    P3 = NgramCharModel(3, wordseq)
    expected_trigrams = {(' ', 't', 'r'): 1, ('t', 'r', 'i'): 1,
                         ('r', 'i', 'g'): 1, ('i', 'g', 'r'): 1,
                         ('g', 'r', 'a'): 1, ('r', 'a', 'm'): 1}

    assert len(P3.dictionary) == len(expected_trigrams)
    for bigram, count in expected_trigrams.items():
        assert bigram in P3.dictionary
        assert P3.dictionary[bigram] == count

    test_string = 'trigram trigram trigram'
    wordseq = words(test_string)
    P3 = NgramCharModel(3, wordseq)
    expected_trigrams = {(' ', 't', 'r'): 3, ('t', 'r', 'i'): 3,
                         ('r', 'i', 'g'): 3, ('i', 'g', 'r'): 3,
                         ('g', 'r', 'a'): 3, ('r', 'a', 'm'): 3}

    assert len(P3.dictionary) == len(expected_trigrams)
    for bigram, count in expected_trigrams.items():
        assert bigram in P3.dictionary
        assert P3.dictionary[bigram] == count


def test_samples():
    story = open_data("EN-text/flatland.txt").read()
    story += open_data("gutenberg.txt").read()
    wordseq = words(story)
    P1 = UnigramWordModel(wordseq)
    P2 = NgramWordModel(2, wordseq)
    P3 = NgramWordModel(3, wordseq)

    s1 = P1.samples(10)
    s2 = P3.samples(10)
    s3 = P3.samples(10)

    assert len(s1.split(' ')) == 10
    assert len(s2.split(' ')) == 10
    assert len(s3.split(' ')) == 10


def test_viterbi_segmentation():
    flatland = open_data("EN-text/flatland.txt").read()
    wordseq = words(flatland)
    P = UnigramWordModel(wordseq)
    text = "itiseasytoreadwordswithoutspaces"

    s, p = viterbi_segment(text, P)
    assert s == [
        'it', 'is', 'easy', 'to', 'read', 'words', 'without', 'spaces']


def test_shift_encoding():
    code = shift_encode("This is a secret message.", 17)

    assert code == 'Kyzj zj r jvtivk dvjjrxv.'


def test_shift_decoding():
    flatland = open_data("EN-text/flatland.txt").read()
    ring = ShiftDecoder(flatland)
    msg = ring.decode('Kyzj zj r jvtivk dvjjrxv.')

    assert msg == 'This is a secret message.'


def test_permutation_decoder():
    gutenberg = open_data("gutenberg.txt").read()
    flatland = open_data("EN-text/flatland.txt").read()

    pd = PermutationDecoder(canonicalize(gutenberg))
    assert pd.decode('aba') in ('ece', 'ete', 'tat', 'tit', 'txt')

    pd = PermutationDecoder(canonicalize(flatland))
    assert pd.decode('aba') in ('ded', 'did', 'ece', 'ele', 'eme', 'ere', 'eve', 'eye', 'iti', 'mom', 'ses', 'tat', 'tit')


def test_rot13_encoding():
    code = rot13('Hello, world!')

    assert code == 'Uryyb, jbeyq!'


def test_rot13_decoding():
    flatland = open_data("EN-text/flatland.txt").read()
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



if __name__ == '__main__':
    pytest.main()
