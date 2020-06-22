import random

import pytest
import nlp

from nlp4e import Rules, Lexicon, Grammar, ProbRules, ProbLexicon, ProbGrammar, E0
from nlp4e import Chart, CYK_parse, subspan, astar_search_parsing, beam_search_parsing

# Clumsy imports because we want to access certain nlp.py globals explicitly, because
# they are accessed by functions within nlp.py

random.seed("aima-python")


def test_rules():
    check = {'A': [['B', 'C'], ['D', 'E']], 'B': [['E'], ['a'], ['b', 'c']]}
    assert Rules(A="B C | D E", B="E | a | b c") == check


def test_lexicon():
    check = {'Article': ['the', 'a', 'an'], 'Pronoun': ['i', 'you', 'he']}
    lexicon = Lexicon(Article="the | a | an", Pronoun="i | you | he")
    assert lexicon == check


def test_grammar():
    rules = Rules(A="B C | D E", B="E | a | b c")
    lexicon = Lexicon(Article="the | a | an", Pronoun="i | you | he")
    grammar = Grammar("Simplegram", rules, lexicon)

    assert grammar.rewrites_for('A') == [['B', 'C'], ['D', 'E']]
    assert grammar.isa('the', 'Article')

    grammar = nlp.E_Chomsky
    for rule in grammar.cnf_rules():
        assert len(rule) == 3


def test_generation():
    lexicon = Lexicon(Article="the | a | an",
                      Pronoun="i | you | he")

    rules = Rules(
        S="Article | More | Pronoun",
        More="Article Pronoun | Pronoun Pronoun"
    )

    grammar = Grammar("Simplegram", rules, lexicon)

    sentence = grammar.generate_random('S')
    for token in sentence.split():
        found = False
        for non_terminal, terminals in grammar.lexicon.items():
            if token in terminals:
                found = True
        assert found


def test_prob_rules():
    check = {'A': [(['B', 'C'], 0.3), (['D', 'E'], 0.7)],
             'B': [(['E'], 0.1), (['a'], 0.2), (['b', 'c'], 0.7)]}
    rules = ProbRules(A="B C [0.3] | D E [0.7]", B="E [0.1] | a [0.2] | b c [0.7]")
    assert rules == check


def test_prob_lexicon():
    check = {'Article': [('the', 0.5), ('a', 0.25), ('an', 0.25)],
             'Pronoun': [('i', 0.4), ('you', 0.3), ('he', 0.3)]}
    lexicon = ProbLexicon(Article="the [0.5] | a [0.25] | an [0.25]",
                          Pronoun="i [0.4] | you [0.3] | he [0.3]")
    assert lexicon == check


def test_prob_grammar():
    rules = ProbRules(A="B C [0.3] | D E [0.7]", B="E [0.1] | a [0.2] | b c [0.7]")
    lexicon = ProbLexicon(Article="the [0.5] | a [0.25] | an [0.25]",
                          Pronoun="i [0.4] | you [0.3] | he [0.3]")
    grammar = ProbGrammar("Simplegram", rules, lexicon)

    assert grammar.rewrites_for('A') == [(['B', 'C'], 0.3), (['D', 'E'], 0.7)]
    assert grammar.isa('the', 'Article')

    grammar = nlp.E_Prob_Chomsky
    for rule in grammar.cnf_rules():
        assert len(rule) == 4


def test_prob_generation():
    lexicon = ProbLexicon(Verb="am [0.5] | are [0.25] | is [0.25]",
                          Pronoun="i [0.4] | you [0.3] | he [0.3]")

    rules = ProbRules(
        S="Verb [0.5] | More [0.3] | Pronoun [0.1] | nobody is here [0.1]",
        More="Pronoun Verb [0.7] | Pronoun Pronoun [0.3]")

    grammar = ProbGrammar("Simplegram", rules, lexicon)

    sentence = grammar.generate_random('S')
    assert len(sentence) == 2


def test_chart_parsing():
    chart = Chart(nlp.E0)
    parses = chart.parses('the stench is in 2 2')
    assert len(parses) == 1


def test_CYK_parse():
    grammar = nlp.E_Prob_Chomsky
    words = ['the', 'robot', 'is', 'good']
    P = CYK_parse(words, grammar)
    assert len(P) == 5

    grammar = nlp.E_Prob_Chomsky_
    words = ['astronomers', 'saw', 'stars']
    P = CYK_parse(words, grammar)
    assert len(P) == 3


def test_subspan():
    spans = subspan(3)
    assert spans.__next__() == (1, 1, 2)
    assert spans.__next__() == (2, 2, 3)
    assert spans.__next__() == (1, 1, 3)
    assert spans.__next__() == (1, 2, 3)


def test_text_parsing():
    words = ["the", "wumpus", "is", "dead"]
    grammer = E0
    assert astar_search_parsing(words, grammer) == 'S'
    assert beam_search_parsing(words, grammer) == 'S'
    words = ["the", "is", "wupus", "dead"]
    assert astar_search_parsing(words, grammer) is False
    assert beam_search_parsing(words, grammer) is False


if __name__ == '__main__':
    pytest.main()
