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
# # NEURAL NETWORKS
#
# This notebook covers the neural network algorithms from chapter 18 of the book *Artificial Intelligence: A Modern Approach*, by Stuart Russel and Peter Norvig. The code in the notebook can be found in [learning.py](https://github.com/aimacode/aima-python/blob/master/learning.py).
#
# Execute the below cell to get started:

# %%
from aima.learning import *

from aima.notebook_utils import psource, pseudocode

# %% [markdown]
# ## NEURAL NETWORK ALGORITHM
#
# ### Overview
#
# Although the Perceptron may seem like a good way to make classifications, it is a linear classifier (which, roughly, means it can only draw straight lines to divide spaces) and therefore it can be stumped by more complex problems. To solve this issue we can extend Perceptron by employing multiple layers of its functionality. The construct we are left with is called a Neural Network, or a Multi-Layer Perceptron, and it is a non-linear classifier. It achieves that by combining the results of linear functions on each layer of the network.
#
# Similar to the Perceptron, this network also has an input and output layer; however, it can also have a number of hidden layers. These hidden layers are responsible for the non-linearity of the network. The layers are comprised of nodes. Each node in a layer (excluding the input one), holds some values, called *weights*, and takes as input the output values of the previous layer. The node then calculates the dot product of its inputs and its weights and then activates it with an *activation function* (e.g. sigmoid activation function). Its output is then fed to the nodes of the next layer. Note that sometimes the output layer does not use an activation function, or uses a different one from the rest of the network. The process of passing the outputs down the layer is called *feed-forward*.
#
# After the input values are fed-forward into the network, the resulting output can be used for classification. The problem at hand now is how to train the network (i.e. adjust the weights in the nodes). To accomplish that we utilize the *Backpropagation* algorithm. In short, it does the opposite of what we were doing up to this point. Instead of feeding the input forward, it will track the error backwards. So, after we make a classification, we check whether it is correct or not, and how far off we were. We then take this error and propagate it backwards in the network, adjusting the weights of the nodes accordingly. We will run the algorithm on the given input/dataset for a fixed amount of time, or until we are satisfied with the results. The number of times we will iterate over the dataset is called *epochs*. In a later section we take a detailed look at how this algorithm works.
#
# NOTE: Sometimes we add another node to the input of each layer, called *bias*. This is a constant value that will be fed to the next layer, usually set to 1. The bias generally helps us "shift" the computed function to the left or right.

# %% [markdown]
# ![neural_net](images/neural_net.png)

# %% [markdown]
# ### Implementation
#
# The `NeuralNetLearner` function takes as input a dataset to train upon, the learning rate (in (0, 1]), the number of epochs and finally the size of the hidden layers. This last argument is a list, with each element corresponding to one hidden layer.
#
# After that we will create our neural network in the `network` function. This function will make the necessary connections between the input layer, hidden layer and output layer. With the network ready, we will use the `BackPropagationLearner` to train the weights of our network for the examples provided in the dataset.
#
# The NeuralNetLearner returns the `predict` function which, in short, can receive an example and feed-forward it into our network to generate a prediction.
#
# In more detail, the example values are first passed to the input layer and then they are passed through the rest of the layers. Each node calculates the dot product of its inputs and its weights, activates it and pushes it to the next layer. The final prediction is the node in the output layer with the maximum value.

# %%
psource(NeuralNetLearner)

# %% [markdown]
# ## BACKPROPAGATION
#
# ### Overview
#
# In both the Perceptron and the Neural Network, we are using the Backpropagation algorithm to train our model by updating the weights. This is achieved by propagating the errors from our last layer (output layer) back to our first layer (input layer), this is why it is called Backpropagation. In order to use Backpropagation, we need a cost function. This function is responsible for indicating how good our neural network is for a given example. One common cost function is the *Mean Squared Error* (MSE). This cost function has the following format:
#
# $$MSE=\frac{1}{n} \sum_{i=1}^{n}(y - \hat{y})^{2}$$
#
# Where `n` is the number of training examples, $\hat{y}$ is our prediction and $y$ is the correct prediction for the example.
#
# The algorithm combines the concept of partial derivatives and the chain rule to generate the gradient for each weight in the network based on the cost function.
#
# For example, if we are using a Neural Network with three layers, the sigmoid function as our activation function and the MSE cost function, we want to find the gradient for the a given weight $w_{j}$, we can compute it like this:
#
# $$\frac{\partial MSE(\hat{y}, y)}{\partial w_{j}} = \frac{\partial MSE(\hat{y}, y)}{\partial \hat{y}}\times\frac{\partial\hat{y}(in_{j})}{\partial in_{j}}\times\frac{\partial in_{j}}{\partial w_{j}}$$
#
# Solving this equation, we have:
#
# $$\frac{\partial MSE(\hat{y}, y)}{\partial w_{j}} = (\hat{y} - y)\times{\hat{y}}'(in_{j})\times a_{j}$$
#
# Remember that $\hat{y}$ is the activation function applied to a neuron in our hidden layer, therefore $$\hat{y} = sigmoid(\sum_{i=1}^{num\_neurons}w_{i}\times a_{i})$$
#
# Also $a$ is the input generated by feeding the input layer variables into the hidden layer.
#
# We can use the same technique for the weights in the input layer as well. After we have the gradients for both weights, we use gradient descent to update the weights of the network.

# %% [markdown]
# ### Pseudocode

# %%
pseudocode('Back-Prop-Learning')

# %% [markdown]
# ### Implementation
#
# First, we feed-forward the examples in our neural network. After that, we calculate the gradient for each layers' weights by using the chain rule. Once that is complete, we update all the weights using gradient descent. After running these for a given number of epochs, the function returns the trained Neural Network.

# %%
psource(BackPropagationLearner)

# %%
iris = DataSet(name="iris")
iris.classes_to_numbers()

nNL = NeuralNetLearner(iris)
print(nNL([5, 3, 1, 0.1]))

# %% [markdown] pycharm={"name": "#%% md\n"}
# The output should be 0, which means the item should get classified in the first class, "setosa". Note that since the algorithm is non-deterministic (because of the random initial weights) the classification might be wrong. Usually though, it should be correct.
#
# To increase accuracy, you can (most of the time) add more layers and nodes. Unfortunately, increasing the number of layers or nodes also increases the computation cost and might result in overfitting.
#
#
