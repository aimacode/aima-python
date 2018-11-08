# References:
#
# https://www.tensorflow.org/guide/low_level_intro
#

# only needed for python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

model = Sequential()
model.add(keras.layers.Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[keras.metrics.sparse_top_k_categorical_accuracy])

model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
myWeights = model.get_weights()
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

model.fit(test_images, test_labels, epochs=1)
model.set_weights(myWeights)

predictions = model.predict(test_images)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=1)
print('prediction =', predictions)

Examples = {
    'MNIST': [test_images, test_labels, model, myWeights]

}






