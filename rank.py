from collections import namedtuple
from BeautifulSoup import BeautifulSoup
import urllib.request
import re
import math

# ______________________________________________________________________________
# Page Ranking

# First entry in list is the base URL, and then following are relative URL pages
examplePagesSet = ["https://en.wikipedia.org/wiki/", "Aesthetics", "Analytic_philosophy",
                   "Ancient_Greek", "Aristotle", "Astrology","Atheism", "Baruch_Spinoza",
                   "Belief", "Betrand Russell", "Confucius", "Consciousness",
                   "Continental Philosophy", "Dialectic", "Eastern_Philosophy",
                   "Epistemology", "Ethics", "Existentialism", "Friedrich_Nietzsche",
                   "Idealism", "Immanuel_Kant", "List_of_political_philosophers", "Logic",
                   "Metaphysics", "Philosophers", "Philosophy", "Philosophy_of_mind", "Physics",
                   "Plato", "Political_philosophy", "Pythagoras", "Rationalism", "Social_philosophy",
                   "Socrates","Subjectivity", "Theology", "Truth", "Western_philosophy"]


def loadPageHTML( addressList ):
    contentDict = {}
    for addr in addressList:
        with urllib.request.urlopen(addr) as response:
            raw_html = response.read().decode('utf-8')
            # Strip the raw html of all unnessecary content. Basically everything that isn't a link or text
            html = stripRawHTML(raw_html)
            contentDict[addr] = html
    return contentDict

def stripRawHTML( raw_html ):
    # TODO: Strip more out of the raw html
    return re.sub("<head>.*?</head>", "", raw_html, flags=re.DOTALL) # remove <head> section

def determineInlinks( page ):
    inlinks = []
    for addr, indexPage in pagesIndex.items():
        if page.address in indexPage.outlinks:
            inlinks.append(addr)
    return inlinks

def findOutlinks( page, handleURLs=None ):
    urls = re.findall(r'href=[\'"]?([^\'" >]+)', pagesContent[page.address])
    if handleURLs:
        urls = handleURLs(urls)
    return urls


# ______________________________________________________________________________
# HITS Helper Functions

def expand_pages( pages ):
    expanded = {}
    for addr,page in pages.items():
        if addr not in expanded:
            expanded[addr] = page
        for inlink in page.inlinks:
            if inlink not in expanded:
                expanded[inlink] = pagesIndex[inlink]
        for outlink in page.outlinks:
            if outlink not in expanded:
                expanded[outlink] = pagesIndex[outlink]
    return expanded

def relevant_pages(query):
    relevant = {}
    for addr,page in pages.items() #
        if query in pagesContent[page.address]:
            relevant[addr] = page
    return relevant

def normalize( pages ):
    """From the pseudocode: Normalize divides each page's score by the sum of
    the squares of all pages' scores (separately for both the authority and hubs scores).
    """
    summed_hub = sum(page.hub**2 for addr,page in pages.items())
    summed_auth = sum(page.authority**2 for addr,page in pages.items())
    for page in pages:
        page.hub /= summed_hub
        page.authority /= summed_auth

def detectConvergence():
    if "hub_history" not in detectConvergence.__dict__:
        detectConvergence.hub_history, detectConvergence.auth_history = [],[]
    # Calculate average deltaHub and average deltaAuth
    curr_hubs.append(page.hub for addr,page in pageIndex.items())
    curr_auths.append(page.authority for addr,page in pageIndex.items())
    aveDeltaHub = sum( abs(x-y) for x,y in zip(curr_hubs, detectConvergence.hub_history[-1]))
    aveDeltaAuth = sum( abs(x-y) for x,y in zip(curr_auths, detectConvergence.auth_history[-1]))
    if aveDeltaHub < 0.1 and aveDeltaAuth < 0.1:
        return True
    else:
        detectConvergence.hub_history.append(curr_hubs)
        detectConvergence.auth_history.append(curr_auths)
        return False

def inlinks( page ):
    if not page.inlinks:
        page.inlinks = determineInlinks(page)
    return page.inlinks

def outlinks( page ):
    if not page.outlinks:
        page.outlinks = findOutlinks(page)
    return page.outlinks


# ______________________________________________________________________________
# HITS Algorithm

pagesContent = {} # maps Page relative or absolute URL/location to page's HTML content
Page = namedtuple('Page', 'address hub authority inlinks outlinks')
pagesIndex = {}

# convergence = detectConvergence()

# def HITS(query): # returns pages with hub and authority numbers
#     pages = expand_pages(relevant_pages(query)) # in order to 'map' faithfully to pseudocode we
#     for p in pages:                             # won't pass the list of pages as an argument
#         p.authority = 1
#         p.hub = 1
#     while True:
#         for p in pages:
#             p.authority = sum(x.hub for x in inlinks(p))  # p.authority ← ∑i Inlinki(p).Hub
#             p.hub = sum(x.authority for x in outlinks(p)) # p.hub ← ∑i Outlinki(p).Authority
#         normalize(pages)
#         if convergence:
#             break
#     return pages
