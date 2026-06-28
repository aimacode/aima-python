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
# # RNN
#
# ## Overview
#
# When human is thinking, they are thinking based on the understanding of previous time steps but not from scratch. Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, imagine you want to do sentimental analysis of some texts. It will be unclear if the traditional network cannot recognize the short phrase and sentences.
#
# Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.
#
# <img src="images/rnn_unit.png" width="500"/>

# %% [markdown]
# A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the above loop:
#  
# <img src="images/rnn_units.png" width="500"/>

# %% [markdown]
# As demonstrated in the book, recurrent neural networks may be connected in many different ways: sequences in the input, the output, or in the most general case both.
#
# <img src="images/rnn_connections.png" width="700"/>

# %% [markdown]
# ## Implementation
#
# In our case, we implemented rnn with modules offered by the package of `keras`. To use `keras` and our module, you must have both `tensorflow` and `keras` installed as a prerequisite. `keras` offered very well defined high-level neural networks API which allows for easy and fast prototyping. `keras` supports many different types of networks such as convolutional and recurrent neural networks as well as user-defined networks. About how to get started with `keras`, please read the [tutorial](https://keras.io/).
#
# To view our implementation of a simple rnn, please use the following code:

# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.deep_learning import *
from aima.notebook_utils import *

# %%
psource(SimpleRNNLearner)

# %% [markdown]
# `train_data` and `val_data` are needed when creating a simple rnn learner. Both attributes take lists of examples and the targets in a tuple. Please note that we build the network by adding layers to a `Sequential()` model which means data are passed through the network one by one. `SimpleRNN` layer is the key layer of rnn which acts the recursive role. Both `Embedding` and `Dense` layers before and after the rnn layer are used to map inputs and outputs to data in rnn form. And the optimizer used in this case is the Adam optimizer.

# %% [markdown]
# ## Example
#
# Here is an example of how we train the rnn network made with `keras`. In this case, we used the IMDB dataset which can be viewed [here](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) in detail. In short, the dataset is consist of movie reviews in text and their labels of sentiment (positive/negative). After loading the dataset we use `keras_dataset_loader` to split it into training, validation and test datasets.

# %%
from keras.datasets import imdb
data = imdb.load_data(num_words=5000)
train, val, test = keras_dataset_loader(data)

# %% [markdown]
# Then we build and train the rnn model for 10 epochs:

# %%
model = SimpleRNNLearner(train, val, epochs=10)

# %% [markdown]
# The accuracy of the training dataset and validation dataset are both over 80% which is very promising. Now let's try on some random examples in the test set:

# %% [markdown]
# ## Autoencoder
#
# Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning. It works by compressing the input into a latent-space representation, to do transformations on the data. 
#
# <img src="images/autoencoder.png" width="800"/>

# %% [markdown]
# Autoencoders are learned automatically from data examples. It means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input and that it does not require any new engineering, only the appropriate training data.
#
# Autoencoders have different architectures for different kinds of data. Here we only provide a simple example of a vanilla encoder, which means they're only one hidden layer in the network:
#
# <img src="images/vanilla.png" width="500"/>
#
# You can view the source code by:

# %%
psource(AutoencoderLearner)

# %% [markdown]
# It shows we added two dense layers to the network structures.
