import pytest
import nlp

from nlp import loadPageHTML, stripRawHTML, findOutlinks, onlyWikipediaURLS
from nlp import expand_pages, relevant_pages, normalize, ConvergenceDetector, getInlinks
from nlp import getOutlinks, Page, determineInlinks, HITS
from nlp import Rules, Lexicon
from nlp import Unigram, Bigram, Trigram
# Clumsy imports because we want to access certain nlp.py globals explicitly, because
# they are accessed by function's within nlp.py

from unittest.mock import patch
from io import BytesIO


def test_ngram_character_count():
    text_string = 'I like programming'

    unigram = Unigram([], characters=True)
    ngrams = unigram.count_ngrams(text_string)
    expected_ngrams = {('l',): 1, ('i',): 3, ('k',): 1, ('e',): 1,
                       ('p',): 1, ('r',): 2, ('o',): 1, ('g',): 2, ('a',): 1, ('m',): 2,
                       ('n',): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams
        assert ngrams[key] == value

    bigram = Bigram([], characters=True)
    ngrams = bigram.count_ngrams(text_string)
    expected_ngrams = {('l', 'i'): 1, ('i', 'k'): 1, ('k', 'e'): 1, ('p', 'r'): 1,
                       ('r', 'o'): 1, ('o', 'g'): 1, ('g', 'r'): 1, ('r', 'a'): 1, ('a', 'm'): 1,
                       ('m', 'm'): 1, ('m', 'i'): 1, ('i', 'n'): 1, ('n', 'g'): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams

    trigram = Trigram([], characters=True)
    ngrams = trigram.count_ngrams(text_string)
    expected_ngrams = {('l', 'i', 'k'): 1, ('i', 'k', 'e'): 1, ('p', 'r', 'o'): 1,
                       ('r', 'o', 'g'): 1, ('o', 'g', 'r'): 1, ('g', 'r', 'a'): 1,
                       ('g', 'r', 'a'): 1, ('r', 'a', 'm'): 1, ('a', 'm', 'm'): 1,
                       ('m', 'm', 'i'): 1, ('m', 'i', 'n'): 1, ('i', 'n', 'g'): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams
        assert ngrams[key] == value
        assert ngrams[key] == value


def test_ngram_word_count():
    text_string = "I like learning about IA"

    unigram = Unigram([])
    ngrams = unigram.count_ngrams(text_string)
    expected_ngrams = {('i',): 1, ('like',): 1, ('learning',): 1,
                       ('about',): 1, ('ia',): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams
        assert ngrams[key] == value

    ngram = Bigram([])
    ngrams = ngram.count_ngrams(text_string)
    expected_ngrams = {('i', 'like'): 1, ('like', 'learning'): 1, ('learning', 'about'): 1,
                       ('about', 'ia'): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams
        assert ngrams[key] == value

    ngram = Trigram([])
    ngrams = ngram.count_ngrams(text_string)
    expected_ngrams = {('i', 'like', 'learning'): 1, ('like', 'learning', 'about'): 1,
                       ('learning', 'about', 'ia'): 1}

    assert len(expected_ngrams) == len(ngrams)

    for key, value in expected_ngrams.items():
        assert key in ngrams
        assert ngrams[key] == value


def test_rules():
    assert Rules(A="B C | D E") == {'A': [['B', 'C'], ['D', 'E']]}


def test_lexicon():
    assert Lexicon(Art="the | a | an") == {'Art': ['the', 'a', 'an']}


# ______________________________________________________________________________
# Data Setup

testHTML = """Keyword String 1: A man is a male human.
            Keyword String 2: Like most other male mammals, a man inherits an
            X from his mom and a Y from his dad.
            Links:
            href="https://google.com.au"
            < href="/wiki/TestThing" > href="/wiki/TestBoy"
            href="/wiki/TestLiving" href="/wiki/TestMan" >"""
testHTML2 = "Nothing"
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

pA = Page("A", 1, 6, ["B", "C", "E"], ["D"])
pB = Page("B", 2, 5, ["E"], ["A", "C", "D"])
pC = Page("C", 3, 4, ["B", "E"], ["A", "D"])
pD = Page("D", 4, 3, ["A", "B", "C", "E"], [])
pE = Page("E", 5, 2, [], ["A", "B", "C", "D", "F"])
pF = Page("F", 6, 1, ["E"], [])
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
    pages = relevant_pages("male")
    assert all((x in pages.keys()) for x in ['A', 'C', 'E'])
    assert all((x not in pages) for x in ['B', 'D', 'F'])


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
