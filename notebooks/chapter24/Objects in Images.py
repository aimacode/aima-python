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
# # Objects in Images
#
# There are two key problems shaping all thinking about objects in images: image classiﬁcation and object detection. They are much more complicated than the problems like boundary detection. Thus more complicated models are needed to deal with the problems even challenging to human's eyes. For the image classification problem, we use a convolutional neural network to extract patterns of an image. For the case of object detection, we use Recursive CNN, which can assist to find the locations of objects of a set of classes in the image. These two models will be detailly introduced in the following sections.

# %% [markdown]
# ## Image Classification
#
# Image classiﬁcation is a task where we decide what class an image of a ﬁxed size belongs to. Traditional ways convert grayscale or RGB images into a list of numbers representing the intensity of that pixel and then do classification job on top of this procedure. Currently One of the most popular techniques used in improving the accuracy of traditional image classification ways is Convolutional Neural Networks which is more similar to the principle of human seeing things.
#
# CNN is different from other neural networks in that it has a convolution layer at the beginning. Instead of converting the image to an array of numbers, the image is broken up into some sections by the convolutional kernel, the machine then tries to predict what each section is. Finally, the computer tries to predict what’s in the picture based on the votes of all sections. 
#
# A classic CNN would has the following architecture:
#
# $$Input ->Convolution ->ReLU ->Convolution ->ReLU ->Pooling -> ... -> Fully Connected$$

# %% [markdown]
# CNNs have an input layer, an output layer, as well as hidden layers. The hidden layers usually consist of convolutional layers, ReLU layers, pooling layers, and fully connected layers. Their functionality can be briefly described as :
#
# - Convolutional layers apply a convolution operation to the input. This layer extracted the features of an image that are used for further processing or classification.
# - Pooling layers combines the outputs of clusters of neurons into a single neuron in the next layer.
# - Fully connected layers connect every neuron in one layer to every neuron in the next layer.
# - RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero.
#
# For a more detailed guidance, please refer to the [course note](http://cs231n.github.io/convolutional-networks/) of Stanford.

# %% [markdown]
# ### Implementation
#
# We implemented a simple CNN with a package of keras which is an advanced level API of TensorFlow. For a more detailed guide, please refer to our previous notebooks or the [official guide](https://keras.io/). The source code can be viewed by importing the necessary packages and executing the following block:

# %%
import os, sys
sys.path = [os.path.abspath("../../")] + sys.path
from aima.perception import *
from aima.notebook_utils import *

# %%
psource(simple_convnet)

# %% [markdown]
# The `simple_convnet` function takes two inputs and returns a Keras `Sequential` model. The input attributes are the number of hidden layers and the number of output classes. One hidden layer is defined as a pair of convolutional layer and max-pooling layer:

# %%
model.add(Conv2D(32, (2, 2), padding='same', kernel_initializer='random_uniform'))
model.add(MaxPooling2D(padding='same'))

# %% [markdown]
# The convolution kernel size we used is of size 2x2 and it is initialized by applying random uniform distribution. We also implemented a helper demonstration function `train_model` to show how the convolutional net performs on a certain dataset. This function only takes a CNN model as input and feeds an MNIST dataset into it. The MNIST dataset is split into the training set, validation set and test set by the number of 1000, 100 and 100.

# %% [markdown]
# ### Example
#
# Now let's try the simple CNN on the MNIST dataset. For the MNIST dataset, there are totally 10 classes: 0-9. Thus we will build a CNN with 10 prediction classes:

# %%
cnn_model = simple_convnet(size=3, num_classes=10)

# %% [markdown]
# The brief description of the CNN architecture is described as above. Please note that each layer has the number of parameters needs to be trained. More parameters meaning longer to train the network on a dataset. We have 3 convolutional layers and 3 max-pooling layers in total and more than 10000 parameters to train.
#
# Now lets train the model for 5 epochs with the pre-defined training parameters: `epochs=5` and `batch_size=32`.

# %%
train_model(cnn_model)

# %% [markdown]
# Within 5 epochs of training, the model accuracy on the training set improves from 35% to 42% while validation accuracy is improved to 46%. This is still relatively low but much higher than the 10% probability of random guess. To improve the accuracy further, you can try both adding more examples to a dataset such as using 20000 training examples and meanwhile training for more rounds.

# %% [markdown]
# ## Object Detection
#
# An object detection program must mark the locations of each object from a known set of classes in test images. Object detection is hard in many aspects: objects can be in various shapes and sometimes maybe deformed or vague. Objects can appear in an image in any position and they are often mixed up with noisy objects or scenes.
#
# Many object detectors are built out of image classiﬁers.On top of the classifier, there is an additional task needed for detecting an object: select objects to be classified with windows and report their precise locations. We usually call windows as bounding boxes and there are multiple ways to build it. The very simplest procedure for choosing windows is to use all windows on some grid. Here we will introduce two main procedures of finding a bounding box.

# %% [markdown]
# ### Selective Search
#
# The simplest procedure for building boxes is to slide a window over the image. It produces a large number of boxes, and the boxes themselves ignore important image evidence but it is designed to be fast. 
#
# Selective Search starts by over-segmenting the image based on the intensity of the pixels using a graph-based segmentation method. Selective Search algorithm takes these segments as initial input and then add all bounding boxes corresponding to segmented parts to the list of regional proposals. Then the algorithm group adjacent segments based on similarity and continue then go repeat the previous steps.
#
#
# #### Implementation
#
# Here we use the selective search method provided by the `opencv-python` package. To use it, please make sure the additional `opencv-contrib-python` version is also installed. You can create a selective search with the following line of code:

# %%
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# %% [markdown]
# Then what to do is to set the input image and selective search mode. Then the model is ready to train:

# %%
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
rects = ss.process()

# %% [markdown]
# The returned `rects` will be the coordinates of the bounding box corners.
#
# #### Example
#
# Here we provided the `selective_search` method to demonstrate the result of the selective search. The method takes a path to the image as input. To execute the demo, please use the following line of code:

# %%
image_path = "./images/stapler.png"
selective_search(image_path)

# %% [markdown]
# The bounding boxes are drawn on the original picture showed in the following:
#
# <img src="images/stapler_bbox.png" width="500"/>

# %% [markdown]
# Some of the bounding boxes do have the stapler or at least most of it in the box, which can assist the classification process.

# %% [markdown]
# ### R-CNN and Faster R-CNN
#
# [Ross Girshick et al.](https://arxiv.org/pdf/1311.2524.pdf) proposed a method where they use selective search to extract just 2000 regions from the image. Then the regions in bounding boxes are feed into a convolutional neural network to perform classification. The brief architecture can be shown as:
#
# <img src="images/RCNN.png" width="500"/>

# %% [markdown]
# The problem with R-CNN is that one must pass each box independently through an image classiﬁer thus it takes a huge amount of time to train the network. And meanwhile, the selective search is not that stable and sometimes may generate bad examples.
#
# Faster R-CNN solved the drawbacks of R-CNN by applying a faster object detection algorithm. Instead of feeding the region proposals to the CNN, we feed the input image to the CNN to generate a convolutional feature map. Then we identify the region of interests on the feature map and then reshape them into a fixed size with an ROI pooling layer so it can be put into another classifier. 
#
# This algorithm is faster than R-CNN as the image is not frequently fed into the CNN to extract feature maps.

# %% [markdown]
# #### Implementation
#
# For an ROI pooling layer, we implemented a simple demo of it as `pool_rois`. We can fake a simple feature map with `numpy`:

# %%
import numpy as np

feature_maps_shape = (200, 100, 1)
feature_map = np.ones(feature_maps_shape, dtype='float32')
feature_map[200 - 1, 100 - 3, 0] = 50

# %% [markdown]
# Note that the fake feature map is all 1 except for one spot with a value of 50. Now let's generate some regio of interests:

# %%
roiss = np.asarray([[0.5, 0.2, 0.7, 0.4], [0.0, 0.0, 1.0, 1.0]])

# %% [markdown]
# Here we only made up two regions of interest. The first only crops some part of the image where all pixels are '1' which ranges from 0.5-0.7 of the length of the horizontal edge and 0.2-0.4 of verticle edge. The range of the second region is the whole image. Now let's pool a 3x7 area out of each region of interest.

# %%
pool_rois(feature_map, roiss, 3, 7)

# %% [markdown]
# What we are expecting is that the second pooled region is different from the first one as there is an artificial feature-the '50' in its input. The printed result is exactly the same as we expected.

# %% [markdown]
# In order to try the whole algorithm of the Faster R-CNN, you can refer to [this GitHub repository](https://github.com/endernewton/tf-faster-rcnn) for more detailed guidance.

# %%
