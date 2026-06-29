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

# %% [markdown]
# ## ENSEMBLE LEARNER
#
# ### Overview
#
# Ensemble Learning improves the performance of our model by combining several learners. It improvise the stability and predictive power of the model. Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to decrease variance, bias, or improve predictions.  
#
#
#
# ![ensemble_learner.jpg](images/ensemble_learner.jpg)
#
#
# Some commonly used Ensemble Learning techniques are : 
#
# 1. Bagging : Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions. It helps us to reduce variance error.
#
# 2. Boosting : Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. It helps us to reduce bias error.
#
# 3.  Stacking : This is a very interesting way of combining models. Here we use a learner to combine output from different learners. It can either decrease bias or variance error depending on the learners we use.
#
# ### Implementation
#
# Below mentioned is the implementation of Ensemble Learner.

# %%
psource(EnsembleLearner)

# %% [markdown]
# This algorithm takes input as a list of learning algorithms, have them vote and then finally returns the predicted result.

# %% [markdown]
# ## AdaBoost
#
# ### Overview
#
# **AdaBoost** is an algorithm which uses **ensemble learning**. In ensemble learning the hypotheses in the collection, or ensemble, vote for what the output should be and the output with the majority votes is selected as the final answer.
#
# AdaBoost algorithm, as mentioned in the book, works with a **weighted training set** and **weak learners** (classifiers that have about 50%+epsilon accuracy i.e slightly better than random guessing). It manipulates the weights attached to the the examples that are showed to it. Importance is given to the examples with higher weights.
#
# All the examples start with equal weights and a hypothesis is generated using these examples. Examples which are incorrectly classified, their weights are increased so that they can be classified correctly by the next hypothesis. The examples that are correctly classified, their weights are reduced. This process is repeated *K* times (here *K* is an input to the algorithm) and hence, *K* hypotheses are generated.
#
# These *K* hypotheses are also assigned weights according to their performance on the weighted training set. The final ensemble hypothesis is the weighted-majority combination of these *K* hypotheses.
#
# The speciality of AdaBoost is that by using weak learners and a sufficiently large *K*, a highly accurate classifier can be learned irrespective of the complexity of the function being learned or the dullness of the hypothesis space.

# %% [markdown]
# ### Implementation
#
# To view the source code of `AdaBoost`, you need to import the necessities first:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.learning import *
from aima.notebook_utils import *
from aima.utils import *

# %% [markdown]
# Then use the following command:

# %%
psource(ada_boost)

# %% [markdown]
# AdaBoost takes as inputs: **L** and *K* where **L** is the learner and *K* is the number of hypotheses to be generated. The learner **L** takes in as inputs: a dataset and the weights associated with the examples in the dataset. But the input learner like `DecisionTreeLearner` doesnot handle weights and only takes a dataset as its input.  
# To remedy that we will give as input to the `DecisionTreeLearner` a modified dataset in which the examples will be repeated according to the weights associated to them. Intuitively, what this will do is force the learner to repeatedly learn the same example again and again until it can classify it correctly.   
#
# To convert `DecisionTreeLearner` so that it can take weights as input too, we will have to pass it through the **`WeightedLearner`** function.

# %%
psource(WeightedLearner)

# %% [markdown]
# The `WeightedLearner` function will then call the `PerceptronLearner`, during each iteration, with the modified dataset which contains the examples according to the weights associated with them.

# %% [markdown]
# ###  Example
#
# We will pass the `DecisionTreeLearner` through `WeightedLearner` function. Then we will create an `AdaboostLearner` classifier with number of hypotheses or *K* equal to 5.

# %%
weighted_tree = WeightedLearner(DecisionTreeLearner)

# %%
iris2 = DataSet(name="iris")
iris2.classes_to_numbers()

adaboost = ada_boost(iris2, weighted_tree, 5)

adaboost([5, 3, 1, 0.1])

# %%
print("Error ratio for adaboost: ", err_ratio(adaboost, iris2))

# %% [markdown]
# Generally using ensemble learning will increase the accuracy of final result as the weight voting of different learners will average the random error.

# %% [markdown]
# ## Evaluate Learners

# %% [markdown]
# We also offer an algorithm evaluating util function: `compare` in the source code. With this function user can compare different algorithms on multiple datasets in order to choose from them.
#
# The default algorithms to compare are `NearestNeighborLearner` and `DecisionTreeLearner`, and the datasets are iris, orings, zoo, restaurant and several other auto-generated test cases.
#
# To use the `compare` function with default settings:

# %%
compare([DecisionTreeLearner, NearestNeighborLearner],
        [DataSet(name='iris'), DataSet(name='orings')])

# %% [markdown]
# As the datasets used here are very simple, there is no significant difference between the error rate of two algorithms except `NearestNeighborLearner` are not doing well on `orings` dataset. You can try self-defined datasets by specifying the `dataset` attribute as the list of datasets of interests such as MNIST.
