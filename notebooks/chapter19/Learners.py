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
# # Learners
#
# In this section, we will introduce several pre-defined learners to learning the datasets by updating their weights to minimize the loss function. when using a learner to deal with machine learning problems, there are several standard steps:
#
# - **Learner initialization**: Before training the network, it usually should be initialized first. There are several choices when initializing the weights: random initialization, initializing weights are zeros or use Gaussian distribution to init the weights.
#
# - **Optimizer specification**: Which means specifying the updating rules of learnable parameters of the network. Usually, we can choose Adam optimizer as default.
#
# - **Applying back-propagation**: In neural networks, we commonly use back-propagation to pass and calculate gradient information of each layer. Back-propagation needs to be integrated with the chosen optimizer in order to update the weights of NN properly in each epoch.
#
# - **Iterations**: Iterating over the forward and back-propagation process of given epochs. Sometimes the iterating process will have to be stopped by triggering early access in case of overfitting.
#
# We will introduce several learners with different structures. We will import all necessary packages before that:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.learning import *
from aima.notebook_utils import *
from aima.deep_learning import *

# %% [markdown]
# ## Perceptron Learner
#
# ### Overview
#
# The Perceptron is a linear classifier. It works the same way as a neural network with no hidden layers (just input and output). First, it trains its weights given a dataset and then it can classify a new item by running it through the network.
#
# Its input layer consists of the item features, while the output layer consists of nodes (also called neurons). Each node in the output layer has *n* synapses (for every item feature), each with its own weight. Then, the nodes find the dot product of the item features and the synapse weights. These values then pass through an activation function (usually a sigmoid). Finally, we pick the largest of the values and we return its index.
#
# Note that in classification problems each node represents a class. The final classification is the class/node with the max output value.
#
# Below you can see a single node/neuron in the outer layer. With *f* we denote the item features, with *w* the synapse weights, then inside the node we have the dot product and the activation function, *g*.

# %% [markdown]
# ![perceptron](images/perceptron.png)

# %% [markdown]
# ### Implementation
#
# Perceptron learner is actually a neural network learner with only one hidden layer which is pre-defined in the algorithm of `perceptron_learner`:

# %%
# input_size and output_size are derived from the dataset
# (here, the iris dataset: 4 features, 3 classes)
input_size, output_size = 4, 3
raw_net = [InputLayer(input_size), DenseLayer(input_size, output_size)]

# %% [markdown]
# Where `input_size` and `output_size` are calculated from dataset examples. In the perceptron learner, the gradient descent optimizer is used to update the weights of the network. we return a function `predict` which we will use in the future to classify a new item. The function computes the (algebraic) dot product of the item with the calculated weights for each node in the outer layer. Then it picks the greatest value and classifies the item in the corresponding class.

# %% [markdown]
# ### Example
#
# Let's try the perceptron learner with the `iris` dataset examples, first let's regulate the dataset classes:

# %%
import numpy as np

iris = DataSet(name="iris")
classes = ["setosa", "versicolor", "virginica"]
iris.classes_to_numbers(classes)
X_iris = np.array([x[:iris.target] for x in iris.examples])
y_iris = np.array([x[iris.target] for x in iris.examples])

# %%
pl = PerceptronLearner(iris, l_rate=0.01, epochs=500, verbose=50).fit(X_iris, y_iris)

# %% [markdown]
# We can see from the printed lines that the final total loss is converged to around 10.50. If we check the error ratio of perceptron learner on the dataset after training, we will see it is much higher than randomly guess:

# %%
print(err_ratio(pl, iris))

# %% [markdown]
# If we test the trained learner with some test cases:

# %%
tests = [([5.0, 3.1, 0.9, 0.1], 0),
        ([5.1, 3.5, 1.0, 0.0], 0),
        ([4.9, 3.3, 1.1, 0.1], 0),
        ([6.0, 3.0, 4.0, 1.1], 1),
        ([6.1, 2.2, 3.5, 1.0], 1),
        ([5.9, 2.5, 3.3, 1.1], 1),
        ([7.5, 4.1, 6.2, 2.3], 2),
        ([7.3, 4.0, 6.1, 2.4], 2),
        ([7.0, 3.3, 6.1, 2.5], 2)]
print(grade_learner(pl, tests))

# %% [markdown]
# It seems the learner is correct on all the test examples.
#
# Now let's try perceptron learner on a more complicated dataset: the MNIST dataset, to see what the result will be. First, we import the dataset to make the examples a `Dataset` object:

# %%
train_img, train_lbl, test_img, test_lbl = load_MNIST(path="../../aima-data/MNIST/Digits")
import numpy as np
import matplotlib.pyplot as plt
train_examples = [np.append(train_img[i], train_lbl[i]) for i in range(len(train_img))]
test_examples = [np.append(test_img[i], test_lbl[i]) for i in range(len(test_img))]
print("length of training dataset:", len(train_examples))
print("length of test dataset:", len(test_examples))

# %% [markdown]
# Now let's train the perceptron learner on the first 1000 examples of the dataset:

# %%
mnist = DataSet(examples=train_examples[:1000])
X_mnist = np.array([x[:mnist.target] for x in mnist.examples])
y_mnist = np.array([x[mnist.target] for x in mnist.examples])
pl = PerceptronLearner(mnist, l_rate=0.01, epochs=10, verbose=1).fit(X_mnist, y_mnist)

# %%
print(err_ratio(pl, mnist))

# %% [markdown]
# It looks like we have a near 90% error ratio on training data after the network is trained on it. Then we can investigate the model's performance on the test dataset which it never has seen before:

# %%
test_mnist = DataSet(examples=test_examples[:100])
print(err_ratio(pl, test_mnist))

# %% [markdown]
# It seems a single layer perceptron learner cannot simulate the structure of the MNIST dataset. To improve accuracy, we may not only increase training epochs but also consider changing to a more complicated network structure.

# %% [markdown]
# ### Neural Network Learner
#
# Although there are many different types of neural networks, the dense neural network we implemented can be treated as a stacked perceptron learner. Adding more layers to the perceptron network could add to the non-linearity to the network thus model will be more flexible when fitting complex data-target relations. Whereas it also adds to the risk of overfitting as the side effect of flexibility.
#
# By default we use dense networks with two hidden layers, which has the architecture as the following:
#
# <img src="images/nn.png" width="500"/>
#
# In our code, we implemented it as:

# %%
# input_size and output_size are derived from the dataset and
# hidden_layer_sizes is the list of hidden layer sizes (here, iris with one hidden layer)
input_size, output_size, hidden_layer_sizes = 4, 3, [4]
# initialize the network
raw_net = [InputLayer(input_size)]
# add hidden layers
hidden_input_size = input_size
for h_size in hidden_layer_sizes:
    raw_net.append(DenseLayer(hidden_input_size, h_size))
    hidden_input_size = h_size
raw_net.append(DenseLayer(hidden_input_size, output_size))

# %% [markdown]
# Where hidden_layer_sizes are the sizes of each hidden layer in a list which can be specified by user. Neural network learner uses gradient descent as default optimizer but user can specify any optimizer when calling `neural_net_learner`. The other special attribute that can be changed in `neural_net_learner` is `batch_size` which controls the number of examples used in each round of update. `neural_net_learner` also returns a `predict` function which calculates prediction by multiplying weight to inputs and applying activation functions.
#
# ### Example
#
# Let's also try `neural_net_learner` on the `iris` dataset:

# %%
nn = NeuralNetworkLearner(iris, [4], l_rate=0.15, epochs=100, optimizer=stochastic_gradient_descent, verbose=10).fit(X_iris, y_iris)

# %% [markdown]
# Similarly we check the model's accuracy on both training and test dataset:

# %%
print("error ration on training set:",err_ratio(nn, iris))

# %%
tests = [([5.0, 3.1, 0.9, 0.1], 0),
        ([5.1, 3.5, 1.0, 0.0], 0),
        ([4.9, 3.3, 1.1, 0.1], 0),
        ([6.0, 3.0, 4.0, 1.1], 1),
        ([6.1, 2.2, 3.5, 1.0], 1),
        ([5.9, 2.5, 3.3, 1.1], 1),
        ([7.5, 4.1, 6.2, 2.3], 2),
        ([7.3, 4.0, 6.1, 2.4], 2),
        ([7.0, 3.3, 6.1, 2.5], 2)]
print("accuracy on test set:",grade_learner(nn, tests))

# %% [markdown]
# We can see that the error ratio on the training set is smaller than the perceptron learner. As the error ratio is relatively small, let's try the model on the MNIST dataset to see whether there will be a larger difference. 

# %%
nn = NeuralNetworkLearner(mnist, [10], l_rate=0.01, epochs=100, optimizer=stochastic_gradient_descent, verbose=10).fit(X_mnist, y_mnist)

# %%
print(err_ratio(nn, mnist))

# %% [markdown]
# After the model converging, the model's error ratio on the training set is still high. We will introduce the convolutional network in the following chapters to see how it helps improve accuracy on learning this dataset.
