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
# # LEARNING APPLICATIONS
#
# In this notebook we will take a look at some indicative applications of machine learning techniques. We will cover content from [`learning.py`](https://github.com/aimacode/aima-python/blob/master/learning.py), for chapter 18 from Stuart Russel's and Peter Norvig's book [*Artificial Intelligence: A Modern Approach*](http://aima.cs.berkeley.edu/). Execute the cell below to get started:

# %%
from aima.learning import *
from aima.probabilistic_learning import *
from aima.notebook_utils import *

# %% [markdown]
# ## CONTENTS
#
# * MNIST Handwritten Digits
#     * Loading and Visualising
#     * Testing
# * MNIST Fashion

# %% [markdown]
# ## MNIST HANDWRITTEN DIGITS CLASSIFICATION
#
# The MNIST Digits database, available from [this page](http://yann.lecun.com/exdb/mnist/), is a large database of handwritten digits that is commonly used for training and testing/validating in Machine learning.
#
# The dataset has **60,000 training images** each of size 28x28 pixels with labels and **10,000 testing images** of size 28x28 pixels with labels.
#
# In this section, we will use this database to compare performances of different learning algorithms.
#
# It is estimated that humans have an error rate of about **0.2%** on this problem. Let's see how our algorithms perform!
#
# NOTE: We will be using external libraries to load and visualize the dataset smoothly ([numpy](http://www.numpy.org/) for loading and [matplotlib](http://matplotlib.org/) for visualization). You do not need previous experience of the libraries to follow along.

# %% [markdown]
# ### Loading MNIST Digits Data
#
# Let's start by loading MNIST data into numpy arrays.
#
# The function `load_MNIST()` loads MNIST data from files saved in `aima-data/MNIST`. It returns four numpy arrays that we are going to use to train and classify hand-written digits in various learning approaches.

# %%
train_img, train_lbl, test_img, test_lbl = load_MNIST()

# %% [markdown]
# Check the shape of these NumPy arrays to make sure we have loaded the database correctly.
#
# Each 28x28 pixel image is flattened to a 784x1 array and we should have 60,000 of them in training data. Similarly, we should have 10,000 of those 784x1 arrays in testing data.

# %%
print("Training images size:", train_img.shape)
print("Training labels size:", train_lbl.shape)
print("Testing images size:", test_img.shape)
print("Testing labels size:", test_lbl.shape)

# %% [markdown]
# ### Visualizing Data
#
# To get a better understanding of the dataset, let's visualize some random images for each class from training and testing datasets.

# %%
# takes 5-10 seconds to execute this
show_MNIST(train_lbl, train_img)

# %%
# takes 5-10 seconds to execute this
show_MNIST(test_lbl, test_img)

# %% [markdown]
# Let's have a look at the average of all the images of training and testing data.

# %%
print("Average of all images in training dataset.")
show_ave_MNIST(train_lbl, train_img)

print("Average of all images in testing dataset.")
show_ave_MNIST(test_lbl, test_img)

# %% [markdown]
# ## Testing
#
# Now, let us convert this raw data into `DataSet.examples` to run our algorithms defined in `learning.py`. Every image is represented by 784 numbers (28x28 pixels) and we append them with its label or class to make them work with our implementations in learning module.

# %%
print(train_img.shape, train_lbl.shape)
temp_train_lbl = train_lbl.reshape((60000,1))
training_examples = np.hstack((train_img, temp_train_lbl))
print(training_examples.shape)

# %% [markdown]
# Now, we will initialize a DataSet with our training examples, so we can use it in our algorithms.

# %%
# takes ~10 seconds to execute this
MNIST_DataSet = DataSet(examples=training_examples, distance=manhattan_distance)

# %% [markdown]
# Moving forward we can use `MNIST_DataSet` to test our algorithms.

# %% [markdown]
# ### Plurality Learner
#
# The Plurality Learner always returns the class with the most training samples. In this case, `1`.

# %%
pL = PluralityLearner(MNIST_DataSet)
print(pL(177))

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[177])
plt.imshow(test_img[177].reshape((28,28)))

# %% [markdown]
# It is obvious that this Learner is not very efficient. In fact, it will guess correctly in only 1135/10000 of the samples, roughly 10%. It is very fast though, so it might have its use as a quick first guess.

# %% [markdown]
# ### Naive-Bayes
#
# The Naive-Bayes classifier is an improvement over the Plurality Learner. It is much more accurate, but a lot slower.

# %%
# takes ~45 Secs. to execute this

nBD = NaiveBayesLearner(MNIST_DataSet, continuous = False)
print(nBD(test_img[0]))

# %% [markdown]
# To make sure that the output we got is correct, let's plot that image along with its label.

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[0])
plt.imshow(test_img[0].reshape((28,28)))

# %% [markdown]
# ### k-Nearest Neighbors
#
# We will now try to classify a random image from the dataset using the kNN classifier.

# %%
# takes ~20 Secs. to execute this
kNN = NearestNeighborLearner(MNIST_DataSet, k=3)
print(kNN(test_img[211]))

# %% [markdown]
# To make sure that the output we got is correct, let's plot that image along with its label.

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[211])
plt.imshow(test_img[211].reshape((28,28)))

# %% [markdown]
# Hurray! We've got it correct. Don't worry if our algorithm predicted a wrong class. With this techinique we have only ~97% accuracy on this dataset.

# %% [markdown]
# ## MNIST FASHION
#
# Another dataset in the same format is [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md). This dataset, instead of digits contains types of apparel (t-shirts, trousers and others). As with the Digits dataset, it is split into training and testing images, with labels from 0 to 9 for each of the ten types of apparel present in the dataset. The below table shows what each label means:
#
# | Label | Description |
# | ----- | ----------- |
# |   0   | T-shirt/top |
# |   1   | Trouser     |
# |   2   | Pullover    |
# |   3   | Dress       |
# |   4   | Coat        |
# |   5   | Sandal      |
# |   6   | Shirt       |
# |   7   | Sneaker     |
# |   8   | Bag         |
# |   9   | Ankle boot  |

# %% [markdown]
# Since both the MNIST datasets follow the same format, the code we wrote for loading and visualizing the Digits dataset will work for Fashion too! The only difference is that we have to let the functions know which dataset we're using, with the `fashion` argument. Let's start by loading the training and testing images:

# %%
train_img, train_lbl, test_img, test_lbl = load_MNIST(fashion=True)

# %% [markdown]
# ### Visualizing Data
#
# Let's visualize some random images for each class, both for the training and testing sections:

# %%
# takes 5-10 seconds to execute this
show_MNIST(train_lbl, train_img, fashion=True)

# %%
# takes 5-10 seconds to execute this
show_MNIST(test_lbl, test_img, fashion=True)

# %% [markdown]
# Let's now see how many times each class appears in the training and testing data:

# %%
print("Average of all images in training dataset.")
show_ave_MNIST(train_lbl, train_img, fashion=True)

print("Average of all images in testing dataset.")
show_ave_MNIST(test_lbl, test_img, fashion=True)

# %% [markdown]
# Unlike Digits, in Fashion all items appear the same number of times.

# %% [markdown]
# ## Testing
#
# We will now begin testing our algorithms on Fashion.
#
# First, we need to convert the dataset into the `learning`-compatible `Dataset` class:

# %%
temp_train_lbl = train_lbl.reshape((60000,1))
training_examples = np.hstack((train_img, temp_train_lbl))

# %%
# takes ~10 seconds to execute this
MNIST_DataSet = DataSet(examples=training_examples, distance=manhattan_distance)

# %% [markdown]
# ### Plurality Learner
#
# The Plurality Learner always returns the class with the most training samples. In this case, `9`.

# %%
pL = PluralityLearner(MNIST_DataSet)
print(pL(177))

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[177])
plt.imshow(test_img[177].reshape((28,28)))

# %% [markdown]
# ### Naive-Bayes
#
# The Naive-Bayes classifier is an improvement over the Plurality Learner. It is much more accurate, but a lot slower.

# %%
# takes ~45 Secs. to execute this

nBD = NaiveBayesLearner(MNIST_DataSet, continuous = False)
print(nBD(test_img[24]))

# %% [markdown]
# Let's check if we got the right output.

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[24])
plt.imshow(test_img[24].reshape((28,28)))

# %% [markdown]
# ### K-Nearest Neighbors
#
# With the dataset in hand, we will first test how the kNN algorithm performs:

# %%
# takes ~20 Secs. to execute this
kNN = NearestNeighborLearner(MNIST_DataSet, k=3)
print(kNN(test_img[211]))

# %% [markdown]
# The output is 1, which means the item at index 211 is a trouser. Let's see if the prediction is correct:

# %%
# %matplotlib inline

print("Actual class of test image:", test_lbl[211])
plt.imshow(test_img[211].reshape((28,28)))

# %% [markdown]
# Indeed, the item was a trouser! The algorithm classified the item correctly.
