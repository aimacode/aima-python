# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run bootstrap.ipynb

# %% [markdown]
# # TEXT
#
# This notebook serves as supporting material for topics covered in **Chapter 22 - Natural Language Processing** from the book *Artificial Intelligence: A Modern Approach*. This notebook uses implementations from [text.py](https://github.com/aimacode/aima-python/blob/master/text.py).

# %%
from aima.text import *
from aima.utils import open_data
from aima.notebook_utils import psource

# %% [markdown]
# ## CONTENTS
#
# * Text Models
# * Viterbi Text Segmentation
# * Information Retrieval
# * Information Extraction
# * Decoders

# %% [markdown]
# ## TEXT MODELS
#
# Before we start analyzing text processing algorithms, we will need to build some language models. Those models serve as a look-up table for character or word probabilities (depending on the type of model). These models can give us the probabilities of words or character sequences appearing in text. Take as example "the". Text models can give us the probability of "the", *P("the")*, either as a word or as a sequence of characters ("t" followed by "h" followed by "e"). The first representation is called "word model" and deals with words as distinct objects, while the second is a "character model" and deals with sequences of characters as objects. Note that we can specify the number of words or the length of the char sequences to better suit our needs. So, given that number of words equals 2, we have probabilities in the form *P(word1, word2)*. For example, *P("of", "the")*. For char models, we do the same but for chars.
#
# It is also useful to store the conditional probabilities of words given preceding words. That means, given we found the words "of" and "the", what is the chance the next word will be "world"? More formally, *P("world"|"of", "the")*. Generalizing, *P(Wi|Wi-1, Wi-2, ... , Wi-n)*.
#
# We call the word model *N-Gram Word Model* (from the Greek "gram", the root of "write", or the word for "letter") and the char model *N-Gram Character Model*. In the special case where *N* is 1, we call the models *Unigram Word Model* and *Unigram Character Model* respectively.
#
# In the `text` module we implement the two models (both their unigram and n-gram variants) by inheriting from the `CountingProbDist` from `learning.py`. Note that `CountingProbDist` does not return the actual probability of each object, but the number of times it appears in our test data.
#
# For word models we have `UnigramWordModel` and `NgramWordModel`. We supply them with a text file and they show the frequency of the different words. We have `UnigramCharModel` and `NgramCharModel` for the character models.
#
# Execute the cells below to take a look at the code.

# %%
psource(UnigramWordModel, NgramWordModel, UnigramCharModel, NgramCharModel)

# %% [markdown]
# Next we build our models. The text file we will use to build them is *Flatland*, by Edwin A. Abbott. We will load it from [here](https://github.com/aimacode/aima-data/blob/a21fc108f52ad551344e947b0eb97df82f8d2b2b/EN-text/flatland.txt). In that directory you can find other text files we might get to use here.

# %% [markdown]
# ### Getting Probabilities
#
# Here we will take a look at how to read text and find the probabilities for each model, and how to retrieve them.
#
# First the word models:

# %%
flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P1 = UnigramWordModel(wordseq)
P2 = NgramWordModel(2, wordseq)

print(P1.top(5))
print(P2.top(5))

print(P1['an'])
print(P2[('i', 'was')])

# %% [markdown]
# We see that the most used word in *Flatland* is 'the', with 2081 occurrences, while the most used sequence is 'of the' with 368 occurrences. Also, the probability of 'an' is approximately 0.003, while for 'i was' it is close to 0.001. Note that the strings used as keys are all lowercase. For the unigram model, the keys are single strings, while for n-gram models we have n-tuples of strings.
#
# Below we take a look at how we can get information from the conditional probabilities of the model, and how we can generate the next word in a sequence.

# %%
flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P3 = NgramWordModel(3, wordseq)

print("Conditional Probabilities Table:", P3.cond_prob[('i', 'was')].dictionary, '\n')
print("Conditional Probability of 'once' give 'i was':", P3.cond_prob[('i', 'was')]['once'], '\n')
print("Next word after 'i was':", P3.cond_prob[('i', 'was')].sample())

# %% [markdown]
# First we print all the possible words that come after 'i was' and the times they have appeared in the model. Next we print the probability of 'once' appearing after 'i was', and finally we pick a word to proceed after 'i was'. Note that the word is picked according to its probability of appearing (high appearance count means higher chance to get picked).

# %% [markdown]
# Let's take a look at the two character models:

# %%
flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P1 = UnigramCharModel(wordseq)
P2 = NgramCharModel(2, wordseq)

print(P1.top(5))
print(P2.top(5))

print(P1['z'])
print(P2[('g', 'h')])

# %% [markdown]
# The most common letter is 'e', appearing more than 19000 times, and the most common sequence is "\_t". That is, a space followed by a 't'. Note that even though we do not count spaces for word models or unigram character models, we do count them for n-gram char models.
#
# Also, the probability of the letter 'z' appearing is close to 0.0006, while for the bigram 'gh' it is 0.003.

# %% [markdown]
# ### Generating Samples
#
# Apart from reading the probabilities for n-grams, we can also use our model to generate word sequences, using the `samples` function in the word models.

# %%
flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P1 = UnigramWordModel(wordseq)
P2 = NgramWordModel(2, wordseq)
P3 = NgramWordModel(3, wordseq)

print(P1.samples(10))
print(P2.samples(10))
print(P3.samples(10))

# %% [markdown]
# For the unigram model, we mostly get gibberish, since each word is picked according to its frequency of appearance in the text, without taking into consideration preceding words. As we increase *n* though, we start to get samples that do have some semblance of conherency and do remind a little bit of normal English. As we increase our data, these samples will get better.
#
# Let's try it. We will add to the model more data to work with and let's see what comes out.

# %%
data = open_data("EN-text/flatland.txt").read()
data += open_data("EN-text/sense.txt").read()

wordseq = words(data)

P3 = NgramWordModel(3, wordseq)
P4 = NgramWordModel(4, wordseq)
P5 = NgramWordModel(5, wordseq)
P7 = NgramWordModel(7, wordseq)

print(P3.samples(15))
print(P4.samples(15))
print(P5.samples(15))
print(P7.samples(15))

# %% [markdown]
# Notice how the samples start to become more and more reasonable as we add more data and increase the *n* parameter. We are still a long way to go though from realistic text generation, but at the same time we can see that with enough data even rudimentary algorithms can output something almost passable.

# %% [markdown]
# ## VITERBI TEXT SEGMENTATION
#
# ### Overview
#
# We are given a string containing words of a sentence, but all the spaces are gone! It is very hard to read and we would like to separate the words in the string. We can accomplish this by employing the `Viterbi Segmentation` algorithm. It takes as input the string to segment and a text model, and it returns a list of the separate words.
#
# The algorithm operates in a dynamic programming approach. It starts from the beginning of the string and iteratively builds the best solution using previous solutions. It accomplishes that by segmentating the string into "windows", each window representing a word (real or gibberish). It then calculates the probability of the sequence up that window/word occurring and updates its solution. When it is done, it traces back from the final word and finds the complete sequence of words.

# %% [markdown]
# ### Implementation

# %%
psource(viterbi_segment)

# %% [markdown]
# The function takes as input a string and a text model, and returns the most probable sequence of words, together with the probability of that sequence.
#
# The "window" is `w` and it includes the characters from *j* to *i*. We use it to "build" the following sequence: from the start to *j* and then `w`. We have previously calculated the probability from the start to *j*, so now we multiply that probability by `P[w]` to get the probability of the whole sequence. If that probability is greater than the probability we have calculated so far for the sequence from the start to *i* (`best[i]`), we update it.

# %% [markdown]
# ### Example
#
# The model the algorithm uses is the `UnigramTextModel`. First we will build the model using the *Flatland* text and then we will try and separate a space-devoid sentence.

# %%
flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)
P = UnigramWordModel(wordseq)
text = "itiseasytoreadwordswithoutspaces"

s, p = viterbi_segment(text,P)
print("Sequence of words is:",s)
print("Probability of sequence is:",p)

# %% [markdown]
# The algorithm correctly retrieved the words from the string. It also gave us the probability of this sequence, which is small, but still the most probable segmentation of the string.

# %% [markdown]
# ## INFORMATION RETRIEVAL
#
# ### Overview
#
# With **Information Retrieval (IR)** we find documents that are relevant to a user's needs for information. A popular example is a web search engine, which finds and presents to a user pages relevant to a query. Information retrieval is not limited only to returning documents though, but can also be used for other type of queries. For example, answering questions when the query is a question, returning information when the query is a concept, and many other applications. An IR system is comprised of the following:
#
# * A body (called corpus) of documents: A collection of documents, where the IR will work on.
#
# * A query language: A query represents what the user wants.
#
# * Results: The documents the system grades as relevant to a user's query and needs.
#
# * Presententation of the results: How the results are presented to the user.
#
# How does an IR system determine which documents are relevant though? We can sign a document as relevant if all the words in the query appear in it, and sign it as irrelevant otherwise. We can even extend the query language to support boolean operations (for example, "paint AND brush") and then sign as relevant the outcome of the query for the document. This technique though does not give a level of relevancy. All the documents are either relevant or irrelevant, but in reality some documents are more relevant than others.
#
# So, instead of a boolean relevancy system, we use a *scoring function*. There are many scoring functions around for many different situations. One of the most used takes into account the frequency of the words appearing in a document, the frequency of a word appearing across documents (for example, the word "a" appears a lot, so it is not very important) and the length of a document (since large documents will have higher occurrences for the query terms, but a short document with a lot of occurrences seems very relevant). We combine these properties in a formula and we get a numeric score for each document, so we can then quantify relevancy and pick the best documents.
#
# These scoring functions are not perfect though and there is room for improvement. For instance, for the above scoring function we assume each word is independent. That is not the case though, since words can share meaning. For example, the words "painter" and "painters" are closely related. If in a query we have the word "painter" and in a document the word "painters" appears a lot, this might be an indication that the document is relevant but we are missing out since we are only looking for "painter". There are a lot of ways to combat this. One of them is to reduce the query and document words into their stems. For example, both "painter" and "painters" have "paint" as their stem form. This can improve slightly the performance of algorithms.
#
# To determine how good an IR system is, we give the system a set of queries (for which we know the relevant pages beforehand) and record the results. The two measures for performance are *precision* and *recall*. Precision measures the proportion of result documents that actually are relevant. Recall measures the proportion of relevant documents (which, as mentioned before, we know in advance) appearing in the result documents.

# %% [markdown]
# ### Implementation
#
# You can read the source code by running the command below:

# %%
psource(IRSystem)

# %% [markdown]
# The `stopwords` argument signifies words in the queries that should not be accounted for in documents. Usually they are very common words that do not add any significant information for a document's relevancy.
#
# A quick guide for the functions in the `IRSystem` class:
#
# * `index_document`: Add document to the collection of documents (named `documents`), which is a list of tuples. Also, count how many times each word in the query appears in each document.
#
# * `index_collection`: Index a collection of documents given by `filenames`.
#
# * `query`: Returns a list of `n` pairs of `(score, docid)` sorted on the score of each document. Also takes care of the special query "learn: X", where instead of the normal functionality we present the output of the terminal command "X".
#
# * `score`: Scores a given document for the given word using `log(1+k)/log(1+n)`, where `k` is the number of query words in a document and `k` is the total number of words in the document. Other scoring functions can be used and you can overwrite this function to better suit your needs.
#
# * `total_score`: Calculate the sum of all the query words in given document.
#
# * `present`/`present_results`: Presents the results as a list.
#
# We also have the class `Document` that holds metadata of documents, like their title, url and number of words. An additional class, `UnixConsultant`, can be used to initialize an IR System for Unix command manuals. This is the example we will use to showcase the implementation.

# %% [markdown]
# ### Example
#
# First let's take a look at the source code of `UnixConsultant`.

# %%
psource(UnixConsultant)

# %% [markdown]
# The class creates an IR System with the stopwords "how do i the a of". We could add more words to exclude, but the queries we will test will generally be in that format, so it is convenient. After the initialization of the system, we get the manual files and start indexing them.
#
# Let's build our Unix consultant and run a query:

# %%
uc = UnixConsultant()

q = uc.query("how do I remove a file")

top_score, top_doc = q[0][0], q[0][1]
print(top_score, uc.documents[top_doc].url)

# %% [markdown]
# We asked how to remove a file and the top result was the `rm` (the Unix command for remove) manual. This is exactly what we wanted! Let's try another query:

# %%
q = uc.query("how do I delete a file")

top_score, top_doc = q[0][0], q[0][1]
print(top_score, uc.documents[top_doc].url)

# %% [markdown]
# Even though we are basically asking for the same thing, we got a different top result. The `diff` command shows the differences between two files. So the system failed us and presented us an irrelevant document. Why is that? Unfortunately our IR system considers each word independent. "Remove" and "delete" have similar meanings, but since they are different words our system will not make the connection. So, the `diff` manual which mentions a lot the word `delete` gets the nod ahead of other manuals, while the `rm` one isn't in the result set since it doesn't use the word at all.

# %% [markdown]
# ## INFORMATION EXTRACTION
#
# **Information Extraction (IE)** is a method for finding occurrences of object classes and relationships in text. Unlike IR systems, an IE system includes (limited) notions of syntax and semantics. While it is difficult to extract object information in a general setting, for more specific domains the system is very useful. One model of an IE system makes use of templates that match with strings in a text.
#
# A typical example of such a model is reading prices from web pages. Prices usually appear after a dollar and consist of numbers, maybe followed by two decimal points. Before the price, usually there will appear a string like "price:". Let's build a sample template.
#
# With the following regular expression (*regex*) we can extract prices from text:
#
# `[$][0-9]+([.][0-9][0-9])?`
#
# Where `+` means 1 or more occurrences and `?` means atmost 1 occurrence. Usually a template consists of a prefix, a target and a postfix regex. In this template, the prefix regex can be "price:", the target regex can be the above regex and the postfix regex can be empty.
#
# A template can match with multiple strings. If this is the case, we need a way to resolve the multiple matches. Instead of having just one template, we can use multiple templates (ordered by priority) and pick the match from the highest-priority template. We can also use other ways to pick. For the dollar example, we can pick the match closer to the numerical half of the highest match. For the text "Price $90, special offer $70, shipping $5" we would pick "$70" since it is closer to the half of the highest match ("$90").

# %% [markdown]
# The above is called *attribute-based* extraction, where we want to find attributes in the text (in the example, the price). A more sophisticated extraction system aims at dealing with multiple objects and the relations between them. When such a system reads the text "$100", it should determine not only the price but also which object has that price.
#
# Relation extraction systems can be built as a series of finite state automata. Each automaton receives as input text, performs transformations on the text and passes it on to the next automaton as input. An automata setup can consist of the following stages:
#
# 1. **Tokenization**: Segments text into tokens (words, numbers and punctuation).
#
# 2. **Complex-word Handling**: Handles complex words such as "give up", or even names like "Smile Inc.".
#
# 3. **Basic-group Handling**: Handles noun and verb groups, segmenting the text into strings of verbs or nouns (for example, "had to give up").
#
# 4. **Complex Phrase Handling**: Handles complex phrases using finite-state grammar rules. For example, "Human+PlayedChess("with" Human+)?" can be one template/rule for capturing a relation of someone playing chess with others.
#
# 5. **Structure Merging**: Merges the structures built in the previous steps.

# %% [markdown]
# Finite-state, template based information extraction models work well for restricted domains, but perform poorly as the domain becomes more and more general. There are many models though to choose from, each with its own strengths and weaknesses. Some of the models are the following:
#
# * **Probabilistic**: Using Hidden Markov Models, we can extract information in the form of prefix, target and postfix from a given text. Two advantages of using HMMs over templates is that we can train HMMs from data and don't need to design elaborate templates, and that a probabilistic approach behaves well even with noise. In a regex, if one character is off, we do not have a match, while with a probabilistic approach we have a smoother process.
#
# * **Conditional Random Fields**: One problem with HMMs is the assumption of state independence. CRFs are very similar to HMMs, but they don't have the latter's constraint. In addition, CRFs make use of *feature functions*, which act as transition weights. For example, if for observation $e_{i}$ and state $x_{i}$ we have $e_{i}$ is "run" and $x_{i}$ is the state ATHLETE, we can have $f(x_{i}, e_{i}) = 1$ and equal to 0 otherwise. We can use multiple, overlapping features, and we can even use features for state transitions. Feature functions don't have to be binary (like the above example) but they can be real-valued as well. Also, we can use any $e$ for the function, not just the current observation. To bring it all together, we weigh a transition by the sum of features.
#
# * **Ontology Extraction**: This is a method for compiling information and facts in a general domain. A fact can be in the form of `NP is NP`, where `NP` denotes a noun-phrase. For example, "Rabbit is a mammal".

# %% [markdown]
# ## DECODERS
#
# ### Introduction
#
# In this section we will try to decode ciphertext using probabilistic text models. A ciphertext is obtained by performing encryption on a text message. This encryption lets us communicate safely, as anyone who has access to the ciphertext but doesn't know how to decode it cannot read the message. We will restrict our study to <b>Monoalphabetic Substitution Ciphers</b>. These are primitive forms of cipher where each letter in the message text (also known as plaintext) is replaced by another another letter of the alphabet.
#
# ### Shift Decoder
#
# #### The Caesar cipher
#
# The Caesar cipher, also known as shift cipher is a form of monoalphabetic substitution ciphers where each letter is <i>shifted</i> by a fixed value. A shift by <b>`n`</b> in this context means that each letter in the plaintext is replaced with a letter corresponding to `n` letters down in the alphabet. For example the plaintext `"ABCDWXYZ"` shifted by `3` yields `"DEFGZABC"`. Note how `X` became `A`. This is because the alphabet is cyclic, i.e. the letter after the last letter in the alphabet, `Z`, is the first letter of the alphabet - `A`.

# %%
plaintext = "ABCDWXYZ"
ciphertext = shift_encode(plaintext, 3)
print(ciphertext)

# %% [markdown]
# #### Decoding a Caesar cipher
#
# To decode a Caesar cipher we exploit the fact that not all letters in the alphabet are used equally. Some letters are used more than others and some pairs of letters are more probable to occur together. We call a pair of consecutive letters a <b>bigram</b>.

# %%
print(bigrams('this is a sentence'))

# %% [markdown]
# We use `CountingProbDist` to get the probability distribution of bigrams. In the latin alphabet consists of only only `26` letters. This limits the total number of possible substitutions to `26`. We reverse the shift encoding for a given `n` and check how probable it is using the bigram distribution. We try all `26` values of `n`, i.e. from `n = 0` to `n = 26` and use the value of `n` which gives the most probable plaintext.

# %%
# %psource ShiftDecoder

# %% [markdown]
# #### Example
#
# Let us encode a secret message using Caeasar cipher and then try decoding it using `ShiftDecoder`. We will again use `flatland.txt` to build the text model

# %%
plaintext = "This is a secret message"
ciphertext = shift_encode(plaintext, 13)
print('The code is', '"' + ciphertext + '"')

# %%
flatland = open_data("EN-text/flatland.txt").read()
decoder = ShiftDecoder(flatland)

decoded_message = decoder.decode(ciphertext)
print('The decoded message is', '"' + decoded_message + '"')

# %% [markdown]
# ### Permutation Decoder
# Now let us try to decode messages encrypted by a general mono-alphabetic substitution cipher. The letters in the alphabet can be replaced by any permutation of letters. For example, if the alphabet consisted of `{A B C}` then it can be replaced by `{A C B}`, `{B A C}`, `{B C A}`, `{C A B}`, `{C B A}` or even `{A B C}` itself. Suppose we choose the permutation `{C B A}`, then the plain text `"CAB BA AAC"` would become `"ACB BC CCA"`. We can see that Caesar cipher is also a form of permutation cipher where the permutation is a cyclic permutation. Unlike the Caesar cipher, it is infeasible to try all possible permutations. The number of possible permutations in Latin alphabet is `26!` which is of the order $10^{26}$. We use graph search algorithms to search for a 'good' permutation.

# %%
psource(PermutationDecoder)

# %% [markdown]
# Each state/node in the graph is represented as a letter-to-letter map. If there is no mapping for a letter, it means the letter is unchanged in the permutation. These maps are stored as dictionaries. Each dictionary is a 'potential' permutation.  We use the word 'potential' because every dictionary doesn't necessarily represent a valid permutation since a permutation cannot have repeating elements. For example the dictionary `{'A': 'B', 'C': 'X'}` is invalid because `'A'` is replaced by `'B'`, but so is `'B'` because the dictionary doesn't have a mapping for `'B'`. Two dictionaries can also represent the same permutation e.g. `{'A': 'C', 'C': 'A'}` and `{'A': 'C', 'B': 'B', 'C': 'A'}` represent the same permutation where `'A'` and `'C'` are interchanged and all other letters remain unaltered. To ensure that we get a valid permutation, a goal state must map all letters in the alphabet. We also prevent repetitions in the permutation by allowing only those actions which go to a new state/node in which the newly added letter to the dictionary maps to previously unmapped letter. These two rules together ensure that the dictionary of a goal state will represent a valid permutation.
# The score of a state is determined using word scores, unigram scores, and bigram scores. Experiment with different weightages for word, unigram and bigram scores and see how they affect the decoding.

# %%
ciphertexts = ['ahed world', 'ahed woxld']

pd = PermutationDecoder(canonicalize(flatland))
for ctext in ciphertexts:
    print('"{}" decodes to "{}"'.format(ctext, pd.decode(ctext)))

# %% [markdown]
# As evident from the above example, permutation decoding using best first search is sensitive to initial text. This is because not only the final dictionary, with substitutions for all letters, must have good score but so must the intermediate dictionaries. You could think of it as performing a local search by finding substitutions for each letter one by one. We could get very different results by changing even a single letter because that letter could be a deciding factor for selecting substitution in early stages which snowballs and affects the later stages. To make the search better we can use different definitions of score in different stages and optimize on which letter to substitute first.
