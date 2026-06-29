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
# # DATASETS

# %% [markdown]
# The following tutorial is a demonstration of the `DataSet` data structure which is frequently used in the following sections. `DataSet` plays the role of organizing data in different forms to make them able to be used by machine learning algorithms. Here we make the following datasets as examples:

# %% [markdown]
# - Fisher's Iris: Each item represents a flower, with four measurements: the length and the width of the sepals and petals. Each item/flower is categorized into one of three species: Setosa, Versicolor and Virginica.
#
# - Zoo: The dataset holds different animals and their classification as "mammal", "fish", etc. The new animal we want to classify has the following measurements: 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1 (don't concern yourself with what the measurements mean).
#
# - Restaurant: The restaurant example in Fig XX of the book. Each item in the dataset represents a condition of customers to make decisions. The target class of each item can be "yes" or "no", meaning whether to dine in this restaurant.
#
# - Orings: The dataset holds different conditions of the night before each launch of the space shuttle. It is to predict the number of O-rings that will experience thermal distress for a given flight when the launch temperature is below freezing. The target class can be 0,1 or 2 meaning the number of oring failures.

# %% [markdown]
# To make use the datasets easier, we have written a class, DataSet, in learning.py. The tutorials found here make use of this class. Now let's have a look at how it works.

# %% [markdown]
# ## Intro

# %% [markdown]
# A lot of the datasets we will work with are .csv files (although other formats are supported too). We have a collection of sample datasets ready to use on [aima-data](https://github.com/aimacode/aima-data/tree/a21fc108f52ad551344e947b0eb97df82f8d2b2b). Four examples are the datasets mentioned above (iris.csv, zoo.csv, orings.csv, and restaurant.csv). You can find plenty of datasets online, and a good repository of such datasets is [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

# %% [markdown]
# In such files, each line corresponds to one item/measurement. Each individual value in a line represents a feature and usually there is a value denoting the class of the item.

# %% [markdown]
# You can find the code for the dataset in `learning.py` or use the following code:

# %%
# %psource DataSet

# %% [markdown]
# ## Importing a Dataset

# %% [markdown]
# There are multiple ways to import a dataset from the `learning` module. But first the necessary modules need to be imported:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.learning import *
from aima.notebook_utils import *

# %% [markdown]
# ### Importing from aima-data

# %% [markdown]
# Dataset uploaded to aima-data can be imported as the following:

# %%
iris = DataSet(name="iris")

# %% [markdown]
# To check that we imported the correct dataset, we can do the following:

# %%
print(iris.examples[0])
print(iris.inputs)

# %% [markdown]
# Which correctly prints the first line in the csv file and the list of attribute indexes.
#
# When importing a dataset, we can specify to exclude an attribute (for example, at index 1) by setting the parameter exclude to the attribute index or name

# %%
iris2 = DataSet(name="iris",exclude=[1])
print(iris2.inputs)


# %% [markdown]
# ### Constructing your own dataset

# %% [markdown]
# In order to use self-defined datasets, you need to prepare the csv files for the datasets in the following format of the [iris example](https://github.com/aimacode/aima-data/blob/a21fc108f52ad551344e947b0eb97df82f8d2b2b/iris.csv). Then you can create your own dataset by specifying the correct dataset name, attributes, targets and exclusive attributes.

# %% [markdown]
# Here is how we create restaurant dataset in Figure 18.3 from restaurant.csv:

# %%
def RestaurantDataSet(examples=None):
    """Build a DataSet of Restaurant waiting examples. [Figure 18.3]"""
    return DataSet(name='restaurant', target='Wait', examples=examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price ' +
                   'Raining Reservation Type WaitEstimate Wait')


# %% [markdown]
# Please note that the dataset name should be the same to the csv file name in order to assist the program finding the correct file.

# %%
restaurant = RestaurantDataSet()
restaurant.inputs

# %% [markdown]
# ## Class Attributes

# %% [markdown]
# Here we will demonstrate the attributes of a `DataSet` object and how they can be utilized. All the attributes can be specified when defining a dataset.

# %% [markdown]
# - <b>examples</b>: Holds the items of the dataset. Each item is a list of values. Could be indexed or sliced.

# %%
iris.examples[:3]

# %% [markdown]
# - **attrs**: The indexes of the features (by default in the range of [0,f), where f is the number of features). For example, item[i] returns the feature at index i of item.
#
# - **attrnames**: An optional list with attribute names. For example, item[s], where s is a feature name, returns the feature of name s in item.
#
# - **target**: The attribute a learning algorithm will try to predict. By default the last attribute.
#
# - **inputs**: This is the indexes of attributes without the target.

# %%
print("attrs:", iris.attrs)
print("attr_names (by default same as attrs):", iris.attr_names)
print("target:", iris.target)
print("inputs:", iris.inputs)

# %% [markdown]
# - **values**: A list of lists which holds the set of possible values for the corresponding attribute/feature. If initially None, it gets computed (by the function setproblem) from the examples.

# %% [markdown]
# For instance if we want to show the possible values of the first attribute:

# %%
print(iris.values[0])

# %% [markdown]
# - **name**: Name of the dataset.

# %%
print("name:", iris.name)

# %% [markdown]
# - **source**: The source of the dataset (url or other). Not used in the code.
#
# - **exclude**: A list of indexes to exclude from inputs. The list can include either attribute indexes (attrs) or names (attrnames).

# %% [markdown]
# ## Helper Functions

# %% [markdown]
# We will now take a look at the auxiliary functions found in the class. These functions help modify a DataSet object to your needs.

# %% [markdown]
# - **sanitize**: Takes as input an example and returns it with non-input (target) attributes replaced by None. Useful for testing. Keep in mind that the example given is not itself sanitized, but instead a sanitized copy is returned.

# %% [markdown]
# Note that the function doesn't actually change the given example; it returns a sanitized copy of it.

# %%
print("Sanitized:",iris.sanitize(iris.examples[0]))
print("Original:",iris.examples[0])

# %% [markdown]
# - **classes_to_numbers**: Maps the class names of a dataset to numbers. If the class names are not given, they are computed from the dataset values. Useful for classifiers that return a numerical value instead of a string.

# %% [markdown]
# For a lot of the classifiers in the book, classes should have numerical values. With this function we are able to map string class names to numbers.

# %%
print("Class of first example:",iris2.examples[0][iris2.target])
iris2.classes_to_numbers()
print("Class of first example:",iris2.examples[0][iris2.target])

# %% [markdown]
# - **remove_examples**: Removes examples containing a given value. Useful for removing examples with missing values, or for removing classes (needed for binary classifiers).

# %% [markdown]
# Currently the iris dataset has three classes, setosa, virginica and versicolor. We want though to convert it to a binary class dataset (a dataset with two classes). The class we want to remove is "virginica". To accomplish that we will utilize the helper function remove_examples.

# %%
iris2 = DataSet(name="iris")

iris2.remove_examples("virginica")
print(iris2.values[iris2.target])

# %% [markdown]
# - **find_means_and_deviations**: find the mean values and deviations of each class in the dataset.

# %% [markdown]
# In the iris example we have three classes, thus both means and deviations have the length of 3.

# %%
means, deviations = iris.find_means_and_deviations()
print(len(means), len(deviations))

# %%
print("Setosa feature means:", means["setosa"])
print("Versicolor mean for first feature:", means["versicolor"][0])

print("Setosa feature deviations:", deviations["setosa"])
print("Virginica deviation for second feature:",deviations["virginica"][1])

# %% [markdown]
# ## Dataset Visualization

# %% [markdown]
# Since the example datasets are used extensively in the code of the book, below we show the common ways to provide a visualized tool that helps in comprehending the dataset and thus how the algorithms work.

# %% [markdown]
# ### Iris Visualization

# %% [markdown]
# We plot the dataset in a 3D space using matplotlib and the function show_iris from notebook.py. The function takes as input three parameters, i, j and k, which are indicises to the iris features, "Sepal Length", "Sepal Width", "Petal Length" and "Petal Width" (0 to 3). By default we show the first three features.

# %%
iris = DataSet(name="iris")

show_iris()
show_iris(0, 1, 3)
show_iris(1, 2, 3)
