import pytest
import nlp

from nlp import loadPageHTML, stripRawHTML, findOutlinks, onlyWikipediaURLS
from nlp import expand_pages, relevant_pages, normalize, ConvergenceDetector, getInlinks
from nlp import getOutlinks, Page, determineInlinks, HITS
from nlp import Rules, Lexicon, Grammar, ProbRules, ProbLexicon, ProbGrammar
from nlp import Chart, CYK_parse
# Clumsy imports because we want to access certain nlp.py globals explicitly, because
# they are accessed by functions within nlp.py

from unittest.mock import patch
from io import BytesIO


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
        More="Pronoun Verb [0.7] | Pronoun Pronoun [0.3]"
    )

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
    assert len(P) == 52


# ______________________________________________________________________________
# Data Setup

testHTML = """Keyword String 1: A man is a male human.
            Keyword String 2: Like most other male mammals, a man inherits an
            X from his mom and a Y from his dad.
            Links:
            href="https://google.com.au"
            < href="/wiki/TestThing" > href="/wiki/TestBoy"
            href="/wiki/TestLiving" href="/wiki/TestMan" >"""
testHTML2 = "a mom and a dad"
testHTML3 = """
            <!DOCTYPE html>
            <html>
            <head>
            <title>Page Title</title>
            </head>
            <body>

            <p>AIMA book</p>

            </body>
            </html>
            """

pA = Page("A", ["B", "C", "E"], ["D"], 1, 6)
pB = Page("B", ["E"], ["A", "C", "D"], 2, 5)
pC = Page("C", ["B", "E"], ["A", "D"], 3, 4)
pD = Page("D", ["A", "B", "C", "E"], [], 4, 3)
pE = Page("E", [], ["A", "B", "C", "D", "F"], 5, 2)
pF = Page("F", ["E"], [], 6, 1)
pageDict = {pA.address: pA, pB.address: pB, pC.address: pC,
            pD.address: pD, pE.address: pE, pF.address: pF}
nlp.pagesIndex = pageDict
nlp.pagesContent ={pA.address: testHTML, pB.address: testHTML2,
                   pC.address: testHTML, pD.address: testHTML2,
                   pE.address: testHTML, pF.address: testHTML2}

# This test takes a long time (> 60 secs)
# def test_loadPageHTML():
#     # first format all the relative URLs with the base URL
#     addresses = [examplePagesSet[0] + x for x in examplePagesSet[1:]]
#     loadedPages = loadPageHTML(addresses)
#     relURLs = ['Ancient_Greek','Ethics','Plato','Theology']
#     fullURLs = ["https://en.wikipedia.org/wiki/"+x for x in relURLs]
#     assert all(x in loadedPages for x in fullURLs)
#     assert all(loadedPages.get(key,"") != "" for key in addresses)


@patch('urllib.request.urlopen', return_value=BytesIO(testHTML3.encode()))
def test_stripRawHTML(html_mock):
    addr = "https://en.wikipedia.org/wiki/Ethics"
    aPage = loadPageHTML([addr])
    someHTML = aPage[addr]
    strippedHTML = stripRawHTML(someHTML)
    assert "<head>" not in strippedHTML and "</head>" not in strippedHTML
    assert "AIMA book" in someHTML and "AIMA book" in strippedHTML


def test_determineInlinks():
    assert set(determineInlinks(pA)) == set(['B', 'C', 'E'])
    assert set(determineInlinks(pE)) == set([])
    assert set(determineInlinks(pF)) == set(['E'])

def test_findOutlinks_wiki():
    testPage = pageDict[pA.address]
    outlinks = findOutlinks(testPage, handleURLs=onlyWikipediaURLS)
    assert "https://en.wikipedia.org/wiki/TestThing" in outlinks
    assert "https://en.wikipedia.org/wiki/TestThing" in outlinks
    assert "https://google.com.au" not in outlinks
# ______________________________________________________________________________
# HITS Helper Functions


def test_expand_pages():
    pages = {k: pageDict[k] for k in ('F')}
    pagesTwo = {k: pageDict[k] for k in ('A', 'E')}
    expanded_pages = expand_pages(pages)
    assert all(x in expanded_pages for x in ['F', 'E'])
    assert all(x not in expanded_pages for x in ['A', 'B', 'C', 'D'])
    expanded_pages = expand_pages(pagesTwo)
    print(expanded_pages)
    assert all(x in expanded_pages for x in ['A', 'B', 'C', 'D', 'E', 'F'])


def test_relevant_pages():
    pages = relevant_pages("his dad")
    assert all((x in pages) for x in ['A', 'C', 'E'])
    assert all((x not in pages) for x in ['B', 'D', 'F'])
    pages = relevant_pages("mom and dad")
    assert all((x in pages) for x in ['A', 'B', 'C', 'D', 'E', 'F'])
    pages = relevant_pages("philosophy")
    assert all((x not in pages) for x in ['A', 'B', 'C', 'D', 'E', 'F'])


def test_normalize():
    normalize(pageDict)
    print(page.hub for addr, page in nlp.pagesIndex.items())
    expected_hub = [1/91**0.5, 2/91**0.5, 3/91**0.5, 4/91**0.5, 5/91**0.5, 6/91**0.5]  # Works only for sample data above
    expected_auth = list(reversed(expected_hub))
    assert len(expected_hub) == len(expected_auth) == len(nlp.pagesIndex)
    assert expected_hub == [page.hub for addr, page in sorted(nlp.pagesIndex.items())]
    assert expected_auth == [page.authority for addr, page in sorted(nlp.pagesIndex.items())]


def test_detectConvergence():
    # run detectConvergence once to initialise history
    convergence = ConvergenceDetector()
    convergence()
    assert convergence()  # values haven't changed so should return True
    # make tiny increase/decrease to all values
    for _, page in nlp.pagesIndex.items():
        page.hub += 0.0003
        page.authority += 0.0004
    # retest function with values. Should still return True
    assert convergence()
    for _, page in nlp.pagesIndex.items():
        page.hub += 3000000
        page.authority += 3000000
    # retest function with values. Should now return false
    assert not convergence()


def test_getInlinks():
    inlnks = getInlinks(pageDict['A'])
    assert sorted(inlnks) == pageDict['A'].inlinks


def test_getOutlinks():
    outlnks = getOutlinks(pageDict['A'])
    assert sorted(outlnks) == pageDict['A'].outlinks


def test_HITS():
    HITS('inherit')
    auth_list = [pA.authority, pB.authority, pC.authority, pD.authority, pE.authority, pF.authority]
    hub_list = [pA.hub, pB.hub, pC.hub, pD.hub, pE.hub, pF.hub]
    assert max(auth_list) == pD.authority
    assert max(hub_list) == pE.hub


if __name__ == '__main__':
    pytest.main()
