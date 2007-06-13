"""Statistical Language Processing tools.  (Chapter 23)
We define Unigram and Ngram text models, use them to generate random text,
and show the Viterbi algorithm for segmentatioon of letters into words.
Then we show a very simple Information Retrieval system, and an example
working on a tiny sample of Unix manual pages."""

from utils import *
from math import log, exp
import re, probability, string, search

class CountingProbDist(probability.ProbDist):
    """A probability distribution formed by observing and counting examples. 
    If P is an instance of this class and o
    is an observed value, then there are 3 main operations:
    p.add(o) increments the count for observation o by 1.
    p.sample() returns a random element from the distribution.
    p[o] returns the probability for o (as in a regular ProbDist)."""

    def __init__(self, observations=[], default=0):
        """Create a distribution, and optionally add in some observations.
        By default this is an unsmoothed distribution, but saying default=1,
        for example, gives you add-one smoothing."""
        update(self, dictionary=DefaultDict(default), needs_recompute=False,
               table=[], n_obs=0)
        for o in observations:
            self.add(o)
        
    def add(self, o):
        """Add an observation o to the distribution."""
        self.dictionary[o] += 1
        self.n_obs += 1
        self.needs_recompute = True

    def sample(self):
        """Return a random sample from the distribution."""
        if self.needs_recompute: self._recompute()
        if self.n_obs == 0:
            return None
        i = bisect.bisect_left(self.table, (1 + random.randrange(self.n_obs),))
        (count, o) = self.table[i]
        return o

    def __getitem__(self, item):
        """Return an estimate of the probability of item."""
        if self.needs_recompute: self._recompute()
        return self.dictionary[item] / self.n_obs

    def __len__(self):
        if self.needs_recompute: self._recompute()
        return self.n_obs

    def top(self, n):
        "Return (count, obs) tuples for the n most frequent observations."
        items = [(v, k) for (k, v) in self.dictionary.items()]
        items.sort(); items.reverse()
        return items[0:n]

    def _recompute(self):
        """Recompute the total count n_obs and the table of entries."""
        n_obs = 0
        table = []
        for (o, count) in self.dictionary.items():
            n_obs += count
            table.append((n_obs, o))
        update(self, n_obs=float(n_obs), table=table, needs_recompute=False)

#______________________________________________________________________________

class UnigramTextModel(CountingProbDist):
    """This is a discrete probability distribution over words, so you
    can add, sample, or get P[word], just like with CountingProbDist.  You can
    also generate a random text n words long with P.samples(n)"""

    def samples(self, n):
        "Return a string of n words, random according to the model."
        return ' '.join([self.sample() for i in range(n)])

class NgramTextModel(CountingProbDist):
    """This is a discrete probability distribution over n-tuples of words.
    You can add, sample or get P[(word1, ..., wordn)]. The method P.samples(n)
    builds up an n-word sequence; P.add_text and P.add_sequence add data."""

    def __init__(self, n, observation_sequence=[]):
        ## In addition to the dictionary of n-tuples, cond_prob is a
        ## mapping from (w1, ..., wn-1) to P(wn | w1, ... wn-1)
        CountingProbDist.__init__(self)
        self.n = n
        self.cond_prob = DefaultDict(CountingProbDist()) 
        self.add_sequence(observation_sequence)

    ## sample, __len__, __getitem__ inherited from CountingProbDist
    ## Note they deal with tuples, not strings, as inputs

    def add(self, ngram):
        """Count 1 for P[(w1, ..., wn)] and for P(wn | (w1, ..., wn-1)"""
        CountingProbDist.add(self, ngram)        
        self.cond_prob[ngram[:-1]].add(ngram[-1])
        
    def add_sequence(self, words):
        """Add each of the tuple words[i:i+n], using a sliding window.
        Prefix some copies of the empty word, '', to make the start work."""
        n = self.n
        words = ['',] * (n-1) + words
        for i in range(len(words)-n):
            self.add(tuple(words[i:i+n]))

    def samples(self, nwords):
        """Build up a random sample of text n words long, using the"""
        n = self.n
        nminus1gram = ('',) * (n-1)
        output = []
        while len(output) < nwords:
            wn = self.cond_prob[nminus1gram].sample()
            if wn:
                output.append(wn)
                nminus1gram = nminus1gram[1:] + (wn,)
            else: ## Cannot continue, so restart.
                nminus1gram = ('',) * (n-1)
        return ' '.join(output)
    
#______________________________________________________________________________


def viterbi_segment(text, P):
    """Find the best segmentation of the string of characters, given the 
    UnigramTextModel P."""
    # best[i] = best probability for text[0:i]
    # words[i] = best word ending at position i
    n = len(text)
    words = [''] + list(text)
    best = [1.0] + [0.0] * n
    ## Fill in the vectors best, words via dynamic programming
    for i in range(n+1):
        for j in range(0, i):
            w = text[j:i]
            if P[w] * best[i - len(w)] >= best[i]:
                best[i] = P[w] * best[i - len(w)]
                words[i] = w
    ## Now recover the sequence of best words
    sequence = []; i = len(words)-1
    while i > 0:
        sequence[0:0] = [words[i]]
        i = i - len(words[i])
    ## Return sequence of best words and overall probability
    return sequence, best[-1]
    

#______________________________________________________________________________


class IRSystem:
    """A very simple Information Retrieval System, as discussed in Sect. 23.2.
    The constructor s = IRSystem('the a') builds an empty system with two 
    stopwords. Next, index several documents with s.index_document(text, url).
    Then ask queries with s.query('query words', n) to retrieve the top n 
    matching documents.  Queries are literal words from the document,
    except that stopwords are ignored, and there is one special syntax:
    The query "learn: man cat", for example, runs "man cat" and indexes it."""

    def __init__(self, stopwords='the a of'):
        """Create an IR System. Optionally specify stopwords."""
        ## index is a map of {word: {docid: count}}, where docid is an int,
        ## indicating the index into the documents list.
        update(self, index=DefaultDict(DefaultDict(0)), 
               stopwords=set(words(stopwords)), documents=[])

    def index_collection(self, filenames):
        "Index a whole collection of files."
        for filename in filenames:
            self.index_document(open(filename).read(), filename)

    def index_document(self, text, url):
        "Index the text of a document."
        ## For now, use first line for title
        title = text[:text.index('\n')].strip()
        docwords = words(text)
        docid = len(self.documents)
        self.documents.append(Document(title, url, len(docwords)))
        for word in docwords:
            if word not in self.stopwords:
                self.index[word][docid] += 1

    def query(self, query_text, n=10):
        """Return a list of n (score, docid) pairs for the best matches.
        Also handle the special syntax for 'learn: command'."""
        if query_text.startswith("learn:"):
            doctext = os.popen(query_text[len("learn:"):], 'r').read()
            self.index_document(doctext, query_text)
            return []
        qwords = [w for w in words(query_text) if w not in self.stopwords]
        shortest = argmin(qwords, lambda w: len(self.index[w]))
        docs = self.index[shortest]
        results = [(sum([self.score(w, d) for w in qwords]), d) for d in docs]
        results.sort(); results.reverse()
        return results[:n]

    def score(self, word, docid):
        "Compute a score for this word on this docid."
        ## There are many options; here we take a very simple approach
        return (math.log(1 + self.index[word][docid])
                / math.log(1 + self.documents[docid].nwords))

    def present(self, results):
        "Present the results as a list."
        for (score, d) in results:
            doc = self.documents[d]
            print "%5.2f|%25s | %s" % (100 * score, doc.url, doc.title[:45])

    def present_results(self, query_text, n=10):
        "Get results for the query and present them."
        self.present(self.query(query_text, n))

class UnixConsultant(IRSystem):
    """A trivial IR system over a small collection of Unix man pages."""
    def __init__(self):
        IRSystem.__init__(self, stopwords="how do i the a of")
        import os
        mandir = '../data/man/'
        man_files = [mandir + f for f in os.listdir(mandir)]
        self.index_collection(man_files)

class Document:
    """Metadata for a document: title and url; maybe add others later."""
    def __init__(self, title, url, nwords):
        update(self, title=title, url=url, nwords=nwords)

def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and
    converting everything to lowercase (to canonicalize).
    >>> words("``EGAD!'' Edgar cried.")
    ['egad', 'edgar', 'cried']
    """
    return reg.findall(text.lower())

def canonicalize(text):
    """Return a canonical text: only lowercase letters and blanks.
    >>> canonicalize("``EGAD!'' Edgar cried.")
    'egad edgar cried'
    """
    return ' '.join(words(text))


#______________________________________________________________________________

## Example application (not in book): decode a cipher.  
## A cipher is a code that substitutes one character for another.
## A shift cipher is a rotation of the letters in the alphabet,
## such as the famous rot13, which maps A to N, B to M, etc.

#### Encoding 

def shift_encode(plaintext, n):
    """Encode text with a shift cipher that moves each letter up by n letters.
    >>> shift_encode('abc z', 1)
    'bcd a'
    """
    return encode(plaintext, alphabet[n:] + alphabet[:n])
    
def rot13(plaintext):
    """Encode text by rotating letters by 13 spaces in the alphabet.
    >>> rot13('hello')
    'uryyb'
    >>> rot13(rot13('hello'))
    'hello'
    """
    return shift_encode(plaintext, 13)

def encode(plaintext, code):
    "Encodes text, using a code which is a permutation of the alphabet."
    from string import maketrans
    trans = maketrans(alphabet + alphabet.upper(), code + code.upper())
    return plaintext.translate(trans)

alphabet = 'abcdefghijklmnopqrstuvwxyz'  
    
def bigrams(text):
    """Return a list of pairs in text (a sequence of letters or words).
    >>> bigrams('this')
    ['th', 'hi', 'is']
    >>> bigrams(['this', 'is', 'a', 'test'])
    [['this', 'is'], ['is', 'a'], ['a', 'test']]
    """
    return [text[i:i+2] for i in range(len(text) - 1)]

#### Decoding a Shift (or Caesar) Cipher

class ShiftDecoder:
    """There are only 26 possible encodings, so we can try all of them,
    and return the one with the highest probability, according to a 
    bigram probability distribution."""
    def __init__(self, training_text):
        training_text = canonicalize(training_text)
        self.P2 = CountingProbDist(bigrams(training_text), default=1)

    def score(self, plaintext):
        "Return a score for text based on how common letters pairs are."
        s = 1.0
        for bi in bigrams(plaintext):
            s = s * self.P2[bi]
        return s
    
    def decode(self, ciphertext):
        "Return the shift decoding of text with the best score."
        return argmax(all_shifts(ciphertext), self.score)

def all_shifts(text):
    "Return a list of all 26 possible encodings of text by a shift cipher."
    return [shift_encode(text, n) for n in range(len(alphabet))]

#### Decoding a General Permutation Cipher

class PermutationDecoder:
    """This is a much harder problem than the shift decoder.  There are 26!
    permutations, so we can't try them all.  Instead we have to search.
    We want to search well, but there are many things to consider:
    Unigram probabilities (E is the most common letter); Bigram probabilities
    (TH is the most common bigram); word probabilities (I and A are the most
    common one-letter words, etc.); etc.
      We could represent a search state as a permutation of the 26 letters,
    and alter the solution through hill climbing.  With an initial guess
    based on unigram probabilities, this would probably fair well. However,
    I chose instead to have an incremental representation. A state is 
    represented as a letter-to-letter map; for example {'z': 'e'} to
    represent that 'z' will be translated to 'e'
    """
    def __init__(self, training_text, ciphertext=None):
        self.Pwords = UnigramTextModel(words(training_text))
        self.P1 = UnigramTextModel(training_text) # By letter
        self.P2 = NgramTextModel(2, training_text) # By letter pair
        if ciphertext:
            return self.decode(ciphertext)

    def decode(self, ciphertext):
        "Search for a decoding of the ciphertext."
        self.ciphertext = ciphertext
        problem = PermutationDecoderProblem(decoder=self)
        return search.best_first_tree_search(problem, self.score)

    def score(self, ciphertext, code):
        """Score is product of word scores, unigram scores, and bigram scores.
        This can get very small, so we use logs and exp."""
        text = decode(ciphertext, code)
        logP = (sum([log(self.Pwords[word]) for word in words(text)]) +
                sum([log(self.P1[c]) for c in text]) +
                sum([log(self.P2[b]) for b in bigrams(text)]))
        return exp(logP)

class PermutationDecoderProblem(search.Problem):
    def __init__(self, initial=None, goal=None, decoder=None):
        self.initial = initial or {}
        self.decoder = decoder

    def successors(self, state):
        ## Find the best 
        p, plainchar = max([(self.decoder.P1[c], c) 
                            for c in alphabet if c not in state])
        succs = [extend(state, plainchar, cipherchar)] #????
        
    def goal_test(self, state):
        "We're done when we get all 26 letters assigned."
        return len(state) >= 26


#______________________________________________________________________________

