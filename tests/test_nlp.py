import pytest
import nlp
from nlp import loadPageHTML, stripRawHTML, determineInlinks, findOutlinks, onlyWikipediaURLS
from nlp import expand_pages, relevant_pages, normalize, ConvergenceDetector, getInlinks
from nlp import getOutlinks, Page, HITS
from nlp import Rules, Lexicon
# Clumsy imports because we want to access certain nlp.py globals explicitly, because
# they are accessed by function's within nlp.py

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

pA = Page("A", 1, 6, ["B","C","E"],["D"])
pB = Page("B", 2, 5, ["E"],["A","C","D"])
pC = Page("C", 3, 4, ["B","E"],["A","D"])
pD = Page("D", 4, 3, ["A","B","C","E"],[])
pE = Page("E", 5, 2, [],["A","B","C","D","F"])
pF = Page("F", 6, 1, ["E"],[])
pageDict = {pA.address:pA,pB.address:pB,pC.address:pC,
            pD.address:pD,pE.address:pE,pF.address:pF}
nlp.pagesIndex = pageDict
nlp.pagesContent ={pA.address:testHTML,pB.address:testHTML2,
              pC.address:testHTML,pD.address:testHTML2,
              pE.address:testHTML,pF.address:testHTML2}

# This test takes a long time (> 60 secs)
# def test_loadPageHTML():
#     # first format all the relative URLs with the base URL
#     addresses = [examplePagesSet[0] + x for x in examplePagesSet[1:]]
#     loadedPages = loadPageHTML(addresses)
#     relURLs = ['Ancient_Greek','Ethics','Plato','Theology']
#     fullURLs = ["https://en.wikipedia.org/wiki/"+x for x in relURLs]
#     assert all(x in loadedPages for x in fullURLs)
#     assert all(loadedPages.get(key,"") != "" for key in addresses)

def test_stripRawHTML():
    addr = "https://en.wikipedia.org/wiki/Ethics"
    aPage = loadPageHTML([addr])
    someHTML = aPage[addr]
    strippedHTML = stripRawHTML(someHTML)
    assert "<head>" not in strippedHTML and "</head>" not in strippedHTML

def test_determineInlinks():
    # TODO
    assert True

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
    pagesTwo = {k: pageDict[k] for k in ('A','E')}
    expanded_pages = expand_pages(pages)
    assert all(x in expanded_pages for x in ['F','E'])
    assert all(x not in expanded_pages for x in ['A','B','C','D'])
    expanded_pages = expand_pages(pagesTwo)
    print(expanded_pages)
    assert all(x in expanded_pages for x in ['A','B','C','D','E','F'])

def test_relevant_pages():
    pages = relevant_pages("male")
    assert all((x in pages.keys()) for x in ['A','C','E'])
    assert all((x not in pages) for x in ['B','D','F'])

def test_normalize():
    normalize( pageDict )
    print(page.hub for addr,page in nlp.pagesIndex.items())
    expected_hub = [1/91,2/91,3/91,4/91,5/91,6/91] # Works only for sample data above
    expected_auth = list(reversed(expected_hub))
    assert len(expected_hub) == len(expected_auth) == len(nlp.pagesIndex)
    assert expected_hub == [page.hub for addr,page in sorted(nlp.pagesIndex.items())]
    assert expected_auth == [page.authority for addr,page in sorted(nlp.pagesIndex.items())]

def test_detectConvergence():
    # run detectConvergence once to initialise history
    convergence = ConvergenceDetector()
    convergence()
    assert convergence() # values haven't changed so should return True
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
    assert sorted([page.address for page in inlnks]) == pageDict['A'].inlinks

def test_getOutlinks():
    outlnks = getOutlinks(pageDict['A'])
    assert sorted([page.address for page in outlnks]) == pageDict['A'].outlinks

def test_HITS():
    # TODO
    assert True # leave for now

if __name__ == '__main__':
    pytest.main()
