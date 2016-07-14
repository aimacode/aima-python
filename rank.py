from collections import namedtuple
import urllib.request
import re

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
                   "Plato", "Political_philosophy", "Pythagoras", "Rationalism","Social_philosophy",
                   "Socrates", "Subjectivity", "Theology", "Truth", "Western_philosophy"]


def loadPageHTML( addressList ):
    contentDict = {}
    for addr in addressList:
        with urllib.request.urlopen(addr) as response:
            raw_html = response.read().decode('utf-8')
            # Strip raw html of unnessecary content. Basically everything that isn't link or text
            html = stripRawHTML(raw_html)
            contentDict[addr] = html
    return contentDict

def initPages( addressList ):
    pages = {}
    for addr in addressList:
        pages[addr] = Page(addr)
    return pages

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

def onlyWikipediaURLS( urls ):
    wikiURLs = [url for url in urls if url.startswith('/wiki/')]
    return ["https://en.wikipedia.org"+url for url in wikiURLs]


# ______________________________________________________________________________
# HITS Helper Functions

def expand_pages( pages ):
    """From Textbook: adds in every page that links to or is linked from one of
    the relevant pages."""
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
    print("pagesContent in function: ", pagesContent)
    for addr, page in pagesIndex.items():
        if query.lower() in pagesContent[addr].lower():
            relevant[addr] = page
    return relevant

def normalize( pages ):
    """From the pseudocode: Normalize divides each page's score by the sum of
    the squares of all pages' scores (separately for both the authority and hubs scores).
    """
    summed_hub = sum(page.hub**2 for _,page in pages.items())
    summed_auth = sum(page.authority**2 for _,page in pages.items())
    for _, page in pages.items():
        page.hub /= summed_hub
        page.authority /= summed_auth

def detectConvergence():
    if "hub_history" not in detectConvergence.__dict__:
        detectConvergence.hub_history, detectConvergence.auth_history = [],[]
    # Calculate average deltaHub and average deltaAuth
    curr_hubs = [page.hub for addr, page in pagesIndex.items()]
    curr_auths = [page.authority for addr, page in pagesIndex.items()]
    if detectConvergence.hub_history and detectConvergence.auth_history:
        diffsHub = [abs(x-y) for x, y in zip(curr_hubs,detectConvergence.hub_history[-1])]
        diffsAuth = [abs(x-y) for x, y in zip(curr_auths,detectConvergence.auth_history[-1])]
        aveDeltaHub  = sum(diffsHub)/float(len(diffsHub))
        aveDeltaAuth = sum(diffsAuth)/float(len(diffsAuth))
        print(aveDeltaHub, " ", aveDeltaAuth)
        if aveDeltaHub < 0.01 and aveDeltaAuth < 0.01:
            return True
    else:
        print("On first iteration")
    detectConvergence.hub_history.append(curr_hubs)
    detectConvergence.auth_history.append(curr_auths)
    return False

def getInlinks( page ):
    if not page.inlinks:
        page.inlinks = determineInlinks(page)
    return page.inlinks

def getOutlinks( page ):
    if not page.outlinks:
        page.outlinks = findOutlinks(page)
    return page.outlinks


# ______________________________________________________________________________
# HITS Algorithm

class Page(object):

    def __init__(self, address, hub=0, authority=0, inlinks=None, outlinks=None):
        self.address = address
        self.hub = hub
        self.authority = authority
        self.inlinks = inlinks
        self.outlinks = outlinks

pagesContent = {} # maps Page relative or absolute URL/location to page's HTML content
pagesIndex = {}
convergence = detectConvergence() # assign function to variable to mimic pseudocode's syntax

def HITS(query): # returns pages with hub and authority numbers
    pages = expand_pages(relevant_pages(query)) # in order to 'map' faithfully to pseudocode we
    for p in pages:                             # won't pass the list of pages as an argument
        p.authority = 1
        p.hub = 1
    while True: # repeat until... convergence
        for p in pages:
            p.authority = sum(x.hub for x in getInlinks(p))  # p.authority ← ∑i Inlinki(p).Hub
            p.hub = sum(x.authority for x in getOutlinks(p)) # p.hub ← ∑i Outlinki(p).Authority
        normalize(pages)
        if convergence:
            break
    return pages
