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
# # Optimization Algorithms
#
# Training a neural network consists of modifying the network’s parameters to minimize the cost function on the training set. In principle, any kind of optimization algorithm could be used. In practice, modern neural networks are almost always trained with some variant of stochastic gradient descent(SGD). Here we will provide two optimization algorithms: SGD and Adam optimizer.
#
# ## Stochastic Gradient Descent
#
# The goal of an optimization algorithm is to find the value of the parameter to make loss function very low. For some types of models, an optimization algorithm might ﬁnd the global minimum value of loss function, but for neural network, the most efficient way to converge loss function to a local minimum is to minimize loss function according to each example.
#
# Gradient descent uses the following update rule to minimize loss function:

# %% [markdown]
# $$\theta^{(t+1)} = \theta^{(t)}-\alpha\nabla_\theta L(\theta^{(t)})$$

# %% [markdown]
# where t is the time step of the algorithm and $\alpha$ is the learning rate. But this rule could be very costly when $L(\theta)$ is defined as a sum across the entire training set. Using SGD can accelerate the learning process as we can use only a batch of examples to update the parameters. 
#
# We implemented the gradient descent algorithm, which can be viewed with the following code:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.deep_learning import *
from aima.notebook_utils import *

# %%
psource(stochastic_gradient_descent)

# %% [markdown]
# There several key elements need to specify when using a `gradient_descent` optimizer:
#
# - **dataset**: A dataset object we used in the previous chapter, such as `iris` and `orings`.
# - **net**: A neural network object which we will cover in the next chapter.
# - **loss**: The loss function used in representing accuracy.
# - **epochs**: How many rounds the training set is used.
# - **l_rate**: learning rate.
# - **batch_size**: The number of examples is used in each update. When very small batch size is used, gradient descent and be treated as SGD.

# %% [markdown]
# ## Adam Optimizer
#
# To mitigate some of the problems caused by the fact that the gradient ignores the second derivatives, some optimization algorithms incorporate the idea of momentum which keeps a running average of the gradients of past mini-batches. Thus Adam optimizer maintains a table saving the previous gradient result.
#
# To view the pseudocode and the implementation, you can use the following codes:

# %%
psource(adam)

# %% [markdown]
# There are several attributes to specify when using Adam optimizer that is different from gradient descent: rho and delta. These parameters determine the percentage of the last iteration is memorized. For more details of how this algorithm work, please refer to the article [here](https://arxiv.org/abs/1412.6980).
#
# In the Stanford course on deep learning for computer vision, the Adam algorithm is suggested as the default optimization method for deep learning applications: 
# >In practice Adam is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. However, it is often also worth trying SGD+Nesterov Momentum as an alternative.

# %% [markdown]
# # Backpropagation
#
# The above algorithms are optimization algorithms: they update parameters like $\theta$ to get smaller loss values. And back-propagation is the method to calculate the gradient for each layer. For complicated models like deep neural networks, the gradients can not be calculated directly as there are enormous array-valued variables.
#
# Fortunately, back-propagation can calculate the gradients briefly which we can interpret as calculating gradients from the last layer to the first which is the inverse process to the forwarding procedure. The derivation of the loss function is passed to previous layers to make them changing toward the direction of minimizing the loss function.

# %% [markdown]
# <img src="images/backprop.png" width="500"/>

# %% [markdown]
# Applying optimizers and back-propagation algorithm together, we can update the weights of a neural network to minimize the loss function with alternatively doing forward and back-propagation process. Here is a figure form [here](https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e) describing how a neural network updates its weights:
#
# <img src="images/nn_steps.png" width="700"></img>

# %% [markdown]
# In our implementation, all the steps are integrated into the optimizer objects. The forward-backward process of passing information through the whole neural network is put into the method `BackPropagation`. You can view the code with:

# %%
psource(BackPropagation)

# %% [markdown]
# The demonstration of optimizers and back-propagation algorithm will be made together with neural network learners.
