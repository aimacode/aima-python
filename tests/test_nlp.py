import pytest
from nlp import *

def test_rules():
	assert Rules(A = "B C | D E") == {'A': [['B', 'C'], ['D', 'E']]}


def test_lexicon():
	assert Lexicon(Art = "the | a | an") == {'Art': ['the', 'a', 'an']}
	