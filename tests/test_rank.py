import pytest
from rank import *

def test_loadPageHTML():
    # first format all the relative URLs with the base URL
    addresses = [examplePagesSet[0] + x for x in examplePagesSet[1:]]
    loadedPages = loadPageHTML(addresses)
    assert ['Ancient_Greek','Ethics','Plato','Theology'] in loadedPages
    assert all(loadedPages.get(key,None) != None for key in addresses)

def test_stripRawHTML():
    addr = "https://en.wikipedia.org/wiki/Ethics"
    aPage = loadPageHTML([addr])
    someHTML = aPage[addr]
    strippedHTML = stripRawHTML(someHTML)
    assert "<head>" not in strippedHTML and "</head>" not in strippedHTML

def test_determineInlinks():
    pass

def test_findOutlinks_wiki():
    testPage = pageDict['A']
    outlinks = findOutlinks(testPage, handleURLs=onlyWikipediaURLS)
    assert "https://en.wikipedia.org/TestThing" in outlinks
    assert "https://en.wikipedia.org/TestThing" in outlinks
    assert "https://google.com.au" not in outlinks 
# ______________________________________________________________________________
# HITS Helper Functions

testHTML = """Keyword String 1: A man is a male human.
            Keyword String 2: Like most other male mammals, a man inherits an
            X from his mom and a Y from his dad.
            Links:
            href="https://google.com.au"
            < href="/wiki/TestThing" > href="/wiki/TestBoy"
            href="/wiki/TestLiving" href="/wiki/TestMan" >"""
testHTML2 = """Nothing"""

pA = Page("A",1,6,["B","C","E"]["D"])
pB = Page("B",2,5,["E"],["A","C","D"])
pC = Page("C",3,4,["B","E"],["A","D"])
pD = Page("D",4,3,["A","B","C","E"],[])
pE = Page("E",5,2,[],["A","B","C","D"])
pF = Page("F",6,1,["E"],[])
pageDict = {pA.address:pA,pB.address:pB,pC.address:pC,
            pD.address:pD,pE.address:pE,pF.address:pF}
pagesContent = pA.address:testHTML,pB.address:testHTML2,
              pC.address:testHTML,pD.address:testHTML2,
              pE.address:testHTML,pF.address:testHTML2}

def test_expand_pages():
    pages = {k: pageDict[k] for k in ('F')}
    pagesTwo = {k: pageDict[k] for k in ('A','E')}
    expanded_pages = expand_pages(pages)
    assert all(x in expanded_pages for x in ['F','E'])
    assert all(x not in expanded_pages for x in ['A','B','C','D'])
    expanded_pages = expand_pages(pagesTwo)
    assert all(x in expanded_pages for x in ['A','B','C','D','E','F'])

def test_relevant_pages():
    pages = relevant_pages("A man is")
    assert all(x in pages for x in ['A','C','E'])
    assert all(x not in pages for x in ['B','D','F'])

def test_normalize():
    normalize( pageDict )
    expected_hub = [1/55,2/55,3/55,4/55,5/55]
    expected_auth = list(reversed(expected_hub))
    assert expected_hub == [page.hub for addr,page in pageDict.items()]
    assert expected_auth == [page.authority for addr,page in pageDict.items()]

def test_detectConverge():
    pass

def test_inlinks():
    inlinks = inlinks(pageDict['A'])
    assert inlinks == pageDict['A'].inlinks

def test_outlinks():
    outlinks = outlinks(pageDict['A'])
    assert outlinks == pageDict['A'].outlinks

def test_HITS():
    pass
