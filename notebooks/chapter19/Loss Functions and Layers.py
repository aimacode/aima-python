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
# # Loss Function
#
# Loss functions evaluate how well specific algorithm models the given data. Commonly loss functions are used to compare the target data and model's prediction. If predictions deviate too much from actual targets, loss function would output a large value. Usually, loss functions can help other optimization functions to improve the accuracy of the model.
#
# However, there’s no one-size-fits-all loss function to algorithms in machine learning. For each algorithm and machine learning projects, specifying certain loss functions could assist the user in getting better model performance. Here we will demonstrate two loss functions: `mse_loss` and `cross_entropy_loss`.

# %% [markdown]
# ## Min Square Error
#
# Min square error(MSE) is the most commonly used loss function in machine learning. The intuition of MSE is straight forward: the distance between two points represents the difference between them. 

# %% [markdown]
# $$MSE = -\sum_i{(y_i-t_i)^2/n}$$

# %% [markdown]
# Where $y_i$ is the prediction of the ith example and $t_i$ is the target of the ith example. And n is the total number of examples.
#
# Below is a plot of an MSE function where the true target value is 100, and the predicted values range between -10,000 to 10,000. The MSE loss (Y-axis) reaches its minimum value at prediction (X-axis) = 100.

# %% [markdown]
# <img src="images/mse_plot.png" width="500"/>

# %% [markdown]
# ## Cross-Entropy
#
# For most deep learning applications, we can get away with just one loss function: cross-entropy loss function. We can think of most deep learning algorithms as learning probability distributions and what we are learning is a distribution of predictions $P(y|x)$ given a series of inputs. 
#
# To associate input examples x with output examples y, the parameters that maximize the likelihood of the training set should be:

# %% [markdown]
# $$\theta^* = argmax_\theta \prod_{i=0}^n p(y^{(i)}/x^{(i)})$$

# %% [markdown]
# Maxmizing the above formula equals to minimizing the negative log form of it:

# %% [markdown]
# $$\theta^* = argmin_\theta -\sum_{i=0}^n logp(y^{(i)}/x^{(i)})$$

# %% [markdown]
# It can be proven that the above formula equals to minimizing MSE loss.
#
# The majority of deep learning algorithms use cross-entropy in some way. Classiﬁers that use deep learning calculate the cross-entropy between categorical distributions over the output class. For a given class, its contribution to the loss is dependent on its probability in the following trend:

# %% [markdown]
# <img src="images/corss_entropy_plot.png" width="500"/>

# %% [markdown]
# ## Examples
#
# First let's import necessary packages.

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.deep_learning import *
from aima.notebook_utils import *

# %% [markdown]
# # Neural Network Layers
#
# Neural networks may be conveniently described using data structures of computational graphs. A computational graph is a directed graph describing how many variables should be computed, with each variable by computed by applying a speciﬁc operation to a set of other variables. 
#
# In our code, we provide class `NNUnit` as the basic structure of a neural network. The structure of `NNUnit` is simple, it only stores the following information:
#
# - **val**: the value of the current node.
# - **parent**: parents of the current node.
# - **weights**: weights between parent nodes and current node. It should be in the same size as parents.
#
# There is another class `Layer` inheriting from `NNUnit`. A `Layer` object holds a list of nodes that represents all the nodes in a layer. It also has a method `forward` to pass a value through the current layer. Here we will demonstrate several pre-defined types of layers in a Neural Network.

# %% [markdown]
# ### Output Layers
#
# Neural networks need specialized output layers for each type of data we might ask them to produce. For many problems, we need to model discrete variables that have k distinct values instead of just binary variables. For example, models of natural language may predict a single word from among of vocabulary of tens of thousands or even more choices. To represent these distributions, we use a softmax layer:

# %% [markdown]
# $$P(y=i|x)=softmax(h(x)^TW+b)_i$$

# %% [markdown]
# where $W$ is matrix of learned weights of output layer $b$ is a vector of learned biases, and the softmax function is:
#
# $$softmax(z_i)=exp(z_i)/\sum_i exp(z_i)$$

# %% [markdown]
# It is simple to create a output layer and feed an example into it:

# %%
layer = OutputLayer(size=4)
example = [1,2,3,4]
print(layer.forward(example))

# %% [markdown]
# The output can be treated like normalized probability when the input of output layer is calculated by probability.

# %% [markdown]
# ### Input Layers
#
# Input layers can be treated like a mapping layer that maps each element of the input vector to each input layer node. The input layer acts as a storage of input vector information which can be used when doing forward propagation.
#
# In our realization of input layers, the size of the input vector and input layer should match.

# %%
layer = InputLayer(size=3)
example = [1,2,3]
print(layer.forward(example))

# %% [markdown]
# ### Hidden Layers
#
# While processing an input vector x of the neural network, it performs several intermediate computations before producing the output y. We can think of these intermediate computations as the state of memory during the execution of a multi-step program. We call the intermediate computations hidden because the data does not specify the values of these variables.
#
# Most neural network hidden layers are based on a linear transformation followed by the application of an elementwise nonlinear function called the activation function g:
#
# $$h=g(W+b)$$
#
# where W is a learned matrix of weights and b is a learned set of bias parameters.
#
# Here we pre-defined several activation functions in `utils.py`: `sigmoid`, `relu`, `elu`, `tanh` and `leaky_relu`. They are all inherited from the `Activation` class. You can get the value of the function or its derivative at a certain point of x:

# %%
s = Sigmoid()
print("Sigmoid at 0:", s.function(0))
print("Deriavation of sigmoid at 0:", s.derivative(0))

# %% [markdown]
# To create a hidden layer object, there are several attributes need to be specified:
#
# - **in_size**: the input vector size of each hidden layer node.
# - **out_size**: the size of the output vector of the hidden layer. Thus each node will hide the weight of the size of (in_size). The weights will be initialized randomly.
# - **activation**: the activation function used for this layer.
#
# Now let's demonstrate how a dense hidden layer works briefly:

# %%
layer = DenseLayer(in_size=4, out_size=3, activation=Sigmoid)
example = [1,2,3,4]
print(layer.forward(example))

# %% [markdown]
# This layer mapped input of size 4 to output of size 3. 

# %% [markdown]
# ### Convolutional Layers
#
# The convolutional layer is similar to the hidden layer except they use a different forward strategy. The convolutional layer takes an input of multiple channels and does convolution on each channel with a pre-defined kernel function. Thus the output of the convolutional layer will still be with the same number of channels. If we image each input as an image, then channels represent its color model such as RGB. The output will still have the same color model as the input.
#
# Now let's try the one-dimensional convolution layer:

# %%
layer = ConvLayer1D(size=3, kernel_size=3)
example = [[1]*3 for _ in range(3)]
print(layer.forward(example))

# %% [markdown]
# Which can be deemed as a one-dimensional image with three channels.

# %% [markdown]
# ### Pooling Layers
#
# Pooling layers can be treated as a special kind of convolutional layer that uses a special kind of kernel to extract a certain value in the kernel region. Here we use max-pooling to report the maximum value in each group.

# %%
layer = MaxPoolingLayer1D(size=3, kernel_size=3)
example = [[1,2,3,4], [2,3,4,1],[3,4,1,2]]
print(layer.forward(example))

# %% [markdown]
# We can see that each time kernel picks up the maximum value in its region.
