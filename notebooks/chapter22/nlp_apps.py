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
import os, sys
sys.path = [os.path.abspath('../../')] + sys.path  # make the aima package importable


# %% [markdown]
# # NATURAL LANGUAGE PROCESSING APPLICATIONS
#
# In this notebook we will take a look at some indicative applications of natural language processing. We will cover content from [`nlp.py`](https://github.com/aimacode/aima-python/blob/master/nlp.py) and [`text.py`](https://github.com/aimacode/aima-python/blob/master/text.py), for chapters 22 and 23 of Stuart Russel's and Peter Norvig's book [*Artificial Intelligence: A Modern Approach*](http://aima.cs.berkeley.edu/).

# %% [markdown]
# ## CONTENTS
#
# * Language Recognition
# * Author Recognition
# * The Federalist Papers
# * Text Classification

# %% [markdown]
# # LANGUAGE RECOGNITION
#
# A very useful application of text models (you can read more on them on the [`text notebook`](https://github.com/aimacode/aima-python/blob/master/text.ipynb)) is categorizing text into a language. In fact, with enough data we can categorize correctly mostly any text. That is because different languages have certain characteristics that set them apart. For example, in German it is very usual for 'c' to be followed by 'h' while in English we see 't' followed by 'h' a lot.
#
# Here we will build an application to categorize sentences in either English or German.
#
# First we need to build our dataset. We will take as input text in English and in German and we will extract n-gram character models (in this case, *bigrams* for n=2). For English, we will use *Flatland* by Edwin Abbott and for German *Faust* by Goethe.
#
# Let's build our text models for each language, which will hold the probability of each bigram occuring in the text.

# %%
from aima.utils import open_data
from aima.text import *

flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P_flatland = NgramCharModel(2, wordseq)

faust = open_data("GE-text/faust.txt").read()
wordseq = words(faust)

P_faust = NgramCharModel(2, wordseq)

# %% [markdown]
# We can use this information to build a *Naive Bayes Classifier* that will be used to categorize sentences (you can read more on Naive Bayes on the [`learning notebook`](https://github.com/aimacode/aima-python/blob/master/learning.ipynb)). The classifier will take as input the probability distribution of bigrams and given a list of bigrams (extracted from the sentence to be classified), it will calculate the probability of the example/sentence coming from each language and pick the maximum.
#
# Let's build our classifier, with the assumption that English is as probable as German (the input is a dictionary with values the text models and keys the tuple `language, probability`):

# %%
from aima.learning import NaiveBayesLearner

dist = {('English', 1): P_flatland, ('German', 1): P_faust}

nBS = NaiveBayesLearner(dist, simple=True)


# %% [markdown]
# Now we need to write a function that takes as input a sentence, breaks it into a list of bigrams and classifies it with the naive bayes classifier from above.
#
# Once we get the text model for the sentence, we need to unravel it. The text models show the probability of each bigram, but the classifier can't handle that extra data. It requires a simple *list* of bigrams. So, if the text model shows that a bigram appears three times, we need to add it three times in the list. Since the text model stores the n-gram information in a dictionary (with the key being the n-gram and the value the number of times the n-gram appears) we need to iterate through the items of the dictionary and manually add them to the list of n-grams.

# %%
def recognize(sentence, nBS, n):
    sentence = sentence.lower()
    wordseq = words(sentence)
    
    P_sentence = NgramCharModel(n, wordseq)
    
    ngrams = []
    for b, p in P_sentence.dictionary.items():
        ngrams += [b]*p
    
    print(ngrams)
    
    return nBS(ngrams)


# %% [markdown]
# Now we can start categorizing sentences.

# %%
recognize("Ich bin ein platz", nBS, 2)

# %%
recognize("Turtles fly high", nBS, 2)

# %%
recognize("Der pelikan ist hier", nBS, 2)

# %%
recognize("And thus the wizard spoke", nBS, 2)

# %% [markdown]
# You can add more languages if you want, the algorithm works for as many as you like! Also, you can play around with *n*. Here we used 2, but other numbers work too (even though 2 suffices). The algorithm is not perfect, but it has high accuracy even for small samples like the ones we used. That is because English and German are very different languages. The closer together languages are (for example, Norwegian and Swedish share a lot of common ground) the lower the accuracy of the classifier.

# %% [markdown]
# ## AUTHOR RECOGNITION
#
# Another similar application to language recognition is recognizing who is more likely to have written a sentence, given text written by them. Here we will try and predict text from Edwin Abbott and Jane Austen. They wrote *Flatland* and *Pride and Prejudice* respectively.
#
# We are optimistic we can determine who wrote what based on the fact that Abbott wrote his novella on much later date than Austen, which means there will be linguistic differences between the two works. Indeed, *Flatland* uses more modern and direct language while *Pride and Prejudice* is written in a more archaic tone containing more sophisticated wording.
#
# Similarly with Language Recognition, we will first import the two datasets. This time though we are not looking for connections between characters, since that wouldn't give that great results. Why? Because both authors use English and English follows a set of patterns, as we show earlier. Trying to determine authorship based on this patterns would not be very efficient.
#
# Instead, we will abstract our querying to a higher level. We will use words instead of characters. That way we can more accurately pick at the differences between their writing style and thus have a better chance at guessing the correct author.
#
# Let's go right ahead and import our data:

# %%
from aima.utils import open_data
from aima.text import *

flatland = open_data("EN-text/flatland.txt").read()
wordseq = words(flatland)

P_Abbott = UnigramWordModel(wordseq, 5)

pride = open_data("EN-text/pride.txt").read()
wordseq = words(pride)

P_Austen = UnigramWordModel(wordseq, 5)

# %% [markdown]
# This time we set the `default` parameter of the model to 5, instead of 0. If we leave it at 0, then when we get a sentence containing a word we have not seen from that particular author, the chance of that sentence coming from that author is exactly 0 (since to get the probability, we multiply all the separate probabilities; if one is 0 then the result is also 0). To avoid that, we tell the model to add 5 to the count of all the words that appear.
#
# Next we will build the Naive Bayes Classifier:

# %%
from aima.learning import NaiveBayesLearner

dist = {('Abbott', 1): P_Abbott, ('Austen', 1): P_Austen}

nBS = NaiveBayesLearner(dist, simple=True)


# %% [markdown]
# Now that we have build our classifier, we will start classifying. First, we need to convert the given sentence to the format the classifier needs. That is, a list of words.

# %%
def recognize(sentence, nBS):
    sentence = sentence.lower()
    sentence_words = words(sentence)
    
    return nBS(sentence_words)


# %% [markdown]
# First we will input a sentence that is something Abbott would write. Note the use of square and the simpler language.

# %%
recognize("the square is mad", nBS)

# %% [markdown]
# The classifier correctly guessed Abbott.
#
# Next we will input a more sophisticated sentence, similar to the style of Austen.

# %%
recognize("a most peculiar acquaintance", nBS)

# %% [markdown]
# The classifier guessed correctly again.
#
# You can try more sentences on your own. Unfortunately though, since the datasets are pretty small, chances are the guesses will not always be correct.

# %% [markdown]
# ## THE FEDERALIST PAPERS
#
# Let's now take a look at a harder problem, classifying the authors of the [Federalist Papers](https://en.wikipedia.org/wiki/The_Federalist_Papers). The *Federalist Papers* are a series of papers written by Alexander Hamilton, James Madison and John Jay towards establishing the United States Constitution.
#
# What is interesting about these papers is that they were all written under a pseudonym, "Publius", to keep the identity of the authors a secret. Only after Hamilton's death, when a list was found written by him detailing the authorship of the papers, did the rest of the world learn what papers each of the authors wrote. After the list was published, Madison chimed in to make a couple of corrections: Hamilton, Madison said, hastily wrote down the list and assigned some papers to the wrong author!
#
# Here we will try and find out who really wrote these mysterious papers.
#
# To solve this we will learn from the undisputed papers to predict the disputed ones. First, let's read the texts from the file:

# %%
from aima.utils import open_data
from aima.text import *

federalist = open_data("EN-text/federalist.txt").read()

# %% [markdown]
# Let's see how the text looks. We will print the first 500 characters:

# %%
federalist[:500]

# %% [markdown]
# It seems that the text file opens with a license agreement, hardly useful in our case. In fact, the license spans 113 words, while there is also a licensing agreement at the end of the file, which spans 3098 words. We need to remove them. To do so, we will first convert the text into words, to make our lives easier.

# %%
wordseq = words(federalist)
wordseq = wordseq[114:-3098]

# %% [markdown]
# Let's now take a look at the first 100 words:

# %%
' '.join(wordseq[:100])

# %% [markdown]
# Much better.
#
# As with any Natural Language Processing problem, it is prudent to do some text pre-processing and clean our data before we start building our model. Remember that all the papers are signed as 'Publius', so we can safely remove that word, since it doesn't give us any information as to the real author.
#
# NOTE: Since we are only removing a single word from each paper, this step can be skipped. We add it here to show that processing the data in our hands is something we should always be considering. Oftentimes pre-processing the data in just the right way is the difference between a robust model and a flimsy one.

# %%
wordseq = [w for w in wordseq if w != 'publius']

# %% [markdown]
# Now we have to separate the text from a block of words into papers and assign them to their authors. We can see that each paper starts with the word 'federalist', so we will split the text on that word.
#
# The disputed papers are the papers from 49 to 58, from 18 to 20 and paper 64. We want to leave these papers unassigned. Also, note that there are two versions of paper 70; both from Hamilton.
#
# Finally, to keep the implementation intuitive, we add a `None` object at the start of the `papers` list to make the list index match up with the paper numbering (for example, `papers[5]` now corresponds to paper no. 5 instead of the paper no.6 in the 0-indexed Python).

# %%
import re

papers = re.split(r'federalist\s', ' '.join(wordseq))
papers = [p for p in papers if p not in ['', ' ']]
papers = [None] + papers

disputed = list(range(49, 58+1)) + [18, 19, 20, 64]
jay, madison, hamilton = [], [], []
for i, p in enumerate(papers):
    if i in disputed or i == 0:
        continue
    
    if 'jay' in p:
        jay.append(p)
    elif 'madison' in p:
        madison.append(p)
    else:
        hamilton.append(p)

len(jay), len(madison), len(hamilton)

# %% [markdown]
# As we can see, from the undisputed papers Jay wrote 4, Madison 17 and Hamilton 51 (+1 duplicate). Let's now build our word models. The Unigram Word Model again will come in handy.

# %%
hamilton = ''.join(hamilton)
hamilton_words = words(hamilton)
P_hamilton = UnigramWordModel(hamilton_words, default=1)

madison = ''.join(madison)
madison_words = words(madison)
P_madison = UnigramWordModel(madison_words, default=1)

jay = ''.join(jay)
jay_words = words(jay)
P_jay = UnigramWordModel(jay_words, default=1)

# %% [markdown]
# Now it is time to build our new Naive Bayes Learner. It is very similar to the one found in `learning.py`, but with an important difference: it doesn't classify an example, but instead returns the probability of the example belonging to each class. This will allow us to not only see to whom a paper belongs to, but also the probability of authorship as well. 
# We will build two versions of Learners, one will multiply probabilities as is and other will add the logarithms of them.
#
# Finally, since we are dealing with long text and the string of probability multiplications is long, we will end up with the results being rounded to 0 due to floating point underflow. To work around this problem we will use the built-in Python library `decimal`, which allows as to set decimal precision to much larger than normal.
#
# Note that the logarithmic learner will compute a negative likelihood since the logarithm of values less than 1 will be negative.
# Thus, the author with the lesser magnitude of proportion is more likely to have written that paper.
#
#

# %%
import random
import decimal
import math
from decimal import Decimal

decimal.getcontext().prec = 100

def precise_product(numbers):
    result = 1
    for x in numbers:
        result *= Decimal(x)
    return result

def log_product(numbers):
    result = 0.0
    for x in numbers:
        result += math.log(x)
    return result

def NaiveBayesLearner(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    Counter distributions and can then be used to find the probability
    of a given item belonging to each class.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the probabilities for each class."""
        def class_prob(target, e):
            attr = attr_dist[target]
            return precise_product([attr[a] for a in e])

        pred = {t: class_prob(t, example) for t in dist.keys()}

        total = sum(pred.values())
        for k, v in pred.items():
            pred[k] = v / total

        return pred

    return predict

def NaiveBayesLearnerLog(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    Counter distributions and can then be used to find the probability
    of a given item belonging to each class. It will compute the likelihood by adding the logarithms of probabilities.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the probabilities for each class."""
        def class_prob(target, e):
            attr = attr_dist[target]
            return log_product([attr[a] for a in e])

        pred = {t: class_prob(t, example) for t in dist.keys()}

        total = -sum(pred.values())
        for k, v in pred.items():
            pred[k] = v/total

        return pred

    return predict



# %% [markdown]
# Next we will build our Learner. Note that even though Hamilton wrote the most papers, that doesn't make it more probable that he wrote the rest, so all the class probabilities will be equal. We can change them if we have some external knowledge, which for this tutorial we do not have.

# %%
dist = {('Madison', 1): P_madison, ('Hamilton', 1): P_hamilton, ('Jay', 1): P_jay}
nBS = NaiveBayesLearner(dist)
nBSL = NaiveBayesLearnerLog(dist)


# %% [markdown]
# As usual, the `recognize` function will take as input a string and after removing capitalization and splitting it into words, will feed it into the Naive Bayes Classifier.

# %%
def recognize(sentence, nBS):
    return nBS(words(sentence.lower()))


# %% [markdown]
# Now we can start predicting the disputed papers:

# %%
print('\nStraightforward Naive Bayes Learner\n')
for d in disputed:
    probs = recognize(papers[d], nBS)
    results = ['{}: {:.4f}'.format(name, probs[(name, 1)]) for name in 'Hamilton Madison Jay'.split()]
    print('Paper No. {}: {}'.format(d, ' '.join(results)))

print('\nLogarithmic Naive Bayes Learner\n')
for d in disputed:
    probs = recognize(papers[d], nBSL)
    results = ['{}: {:.6f}'.format(name, probs[(name, 1)]) for name in 'Hamilton Madison Jay'.split()]
    print('Paper No. {}: {}'.format(d, ' '.join(results)))



# %% [markdown]
# We can see that both learners classify the papers identically. Because of underflow in the straightforward learner, only one author remains with a positive value. The log learner is more accurate with marginal differences between all the authors. 
#
# This is a simple approach to the problem and thankfully researchers are fairly certain that papers 49-58 were all written by Madison, while 18-20 were written in collaboration between Hamilton and Madison, with Madison being credited for most of the work. Our classifier is not that far off. It correctly identifies the papers written by Madison, even the ones in collaboration with Hamilton.
#
# Unfortunately, it misses paper 64. Consensus is that the paper was written by John Jay, while our classifier believes it was written by Hamilton. The classifier is wrong there because it does not have much information on Jay's writing; only 4 papers. This is one of the problems with using unbalanced datasets such as this one, where information on some classes is sparser than information on the rest. To avoid this, we can add more writings for Jay and Madison to end up with an equal amount of data for each author.

# %% [markdown]
# ## Text Classification

# %% [markdown]
# **Text Classification** is assigning a category to a document based on the content of the document. Text Classification is one of the most popular and fundamental tasks of Natural Language Processing. Text classification can be applied on a variety of texts like *Short Documents* (like tweets, customer reviews, etc.) and *Long Document* (like emails, media articles, etc.).
#
# We already have seen an example of Text Classification in the above tasks like Language Identification, Author Recognition and Federalist Paper Identification.
#
# ### Applications
# Some of the broad applications of Text Classification are:-
# - Language Identification
# - Author Recognition
# - Sentiment Analysis
# - Spam Mail Detection
# - Topic Labelling 
# - Word Sense Disambiguation
#
# ### Use Cases
# Some of the use cases of Text classification are:-
# - Social Media Monitoring
# - Brand Monitoring
# - Auto-tagging of user queries
#
# For Text Classification, we would be using the Naive Bayes Classifier. The reasons for using Naive Bayes Classifier are:-
# - Being a probabilistic classifier, therefore, will calculate the probability of each category
# - It is fast, reliable and accurate 
# - Naive Bayes Classifiers have already been used to solve many Natural Language Processing (NLP) applications.
#
# Here we would here be covering an example of **Word Sense Disambiguation** as an application of Text Classification. It is used to remove the ambiguity of a given word if the word has two different meanings.
#
# As we know that we would be working on determining whether the word *apple* in a sentence refers to `fruit` or to a `company`.

# %% [markdown]
# **Step 1:- Defining the dataset** 
#
# The dataset has been defined here so that everything is clear and can be tested with other things as well.

# %%
train_data = [
    "Apple targets big business with new iOS 7 features. Finally... A corp iTunes account!",
    "apple inc is searching for people to help and try out all their upcoming tablet within our own net page No.",
    "Microsoft to bring Xbox and PC games to Apple, Android phones: Report: Microsoft Corp",
    "When did green skittles change from lime to green apple?",
    "Myra Oltman is the best. I told her I wanted to learn how to make apple pie, so she made me a kit!",
    "Surreal Sat in a sewing room, surrounded by crap, listening to beautiful music eating apple pie."
]

train_target = [
    "company",
    "company",
    "company",
    "fruit",
    "fruit",
    "fruit",
]

class_0 = "company"
class_1 = "fruit"

test_data = [
    "Apple Inc. supplier Foxconn demos its own iPhone-compatible smartwatch",
    "I now know how to make a delicious apple pie thanks to the best teachers ever"
]

# %% [markdown]
# **Step 2:- Preprocessing the dataset**
#
# In this step, we would be doing some preprocessing on the dataset like breaking the sentence into words and converting to lower case.
#
# We already have a `words(sent)` function defined in `text.py` which does the task of splitting the sentence into words.

# %%
train_data_processed = [words(i) for i in train_data]

# %% [markdown]
# **Step 3:- Feature Extraction from the text**
#
# Now we would be extracting features from the text like extracting the set of words used in both the categories i.e. `company` and `fruit`.
#
# The frequency of a word would help in calculating the probability of that word being in a particular class. 

# %%
words_0 = []
words_1 = []

for sent, tag in zip(train_data_processed, train_target):
    if(tag == class_0):
        words_0 += sent
    elif(tag == class_1):
        words_1 += sent
    
print("Number of words in `{}` class: {}".format(class_0, len(words_0)))
print("Number of words in `{}` class: {}".format(class_1, len(words_1)))

# %% [markdown]
# As you might have observed, that our dataset is equally balanced, i.e. we have an equal number of words in both the classes.

# %% [markdown]
# **Step 4:- Building the Naive Bayes Model**
#
# Using the Naive Bayes classifier we can calculate the probability of a word in `company` and `fruit` class and then multiplying all of them to get the probability of that sentence belonging each of the given classes. But if a word is not in our dictionary then this leads to the probability of that word belonging to that class becoming zero. For example:- the word *Foxconn* is not in the dictionary of any of the classes. Due to this, the probability of word *Foxconn* being in any of these classes becomes zero, and since all the probabilities are multiplied, this leads to the probability of that sentence belonging to any of the classes becoming zero.   
#
# To solve the problem we need to use **smoothing**, i.e. providing a minimum non-zero threshold probability to every word that we come across.
#
# The `UnigramWordModel` class has implemented smoothing by taking an additional argument from the user, i.e. the minimum frequency that we would be giving to every word even if it is new to the dictionary.

# %%
model_words_0 = UnigramWordModel(words_0, 1)
model_words_1 = UnigramWordModel(words_1, 1)

# %% [markdown]
# Now we would be building the Naive Bayes model. For that, we would be making `dist` as we had done earlier in the Authorship Recognition Task.

# %%
from aima.learning import NaiveBayesLearner

dist = {('company', 1): model_words_0, ('fruit', 1): model_words_1}

nBS = NaiveBayesLearner(dist, simple=True)


# %% [markdown]
# **Step 5:- Predict the class of a sentence**
#
# Now we will be writing a function that does pre-process of the sentences which we have taken for testing. And then predicting the class of every sentence in the document.

# %%
def recognize(sentence, nBS):
    sentence_words = words(sentence)
    return nBS(sentence_words)


# %%
# predicting the class of sentences in the test set
for i in test_data:
    print(i + "\t-" + recognize(i, nBS))

# %% [markdown]
# You might have observed that the predictions made by the model are correct and we are able to differentiate between sentences of different classes. You can try more sentences on your own. Unfortunately though, since the datasets are pretty small, chances are the guesses will not always be correct.
#
# As you might have observed, the above method is very much similar to the Author Recognition, which is also a type of Text Classification. Like this most of Text Classification have the same underlying structure and follow a similar procedure.
