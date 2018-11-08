# References:
#
# https://www.tensorflow.org/guide/low_level_intro
#

# only needed for python 2.7
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
from numpy import array
from numpy import float32

# this takes a looong time to index, and
# python may crash several times before indexing is complete
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images = train_images / 255.0
# test_images = test_images / 255.0


x_train = np.reshape(x_train, (x_train.shape[0], (x_train.shape[1] * x_train.shape[2])))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
x_test = np.reshape(x_test, (x_test.shape[0], (x_test.shape[1] * x_test.shape[2])))
y_test = np.reshape(y_test, (y_test.shape[0], 1))


# model = Sequential()
# model.add(Dense(8,
#                 activation=keras.activations.sigmoid,
#                 ))
# model.add(Dense(3,
#                 activation=keras.activations.sigmoid,
#                 ))

# model.compile(
#     optimizer=tf.train.AdamOptimizer(0.001),
#     loss=keras.losses.categorical_crossentropy,
#     # loss=keras.losses.mse,
#     # metrics=[keras.metrics.binary_accuracy],
#     metrics=[keras.metrics.categorical_accuracy]
# )

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = Sequential()
tf.keras.layers.Flatten(input_shape=(28, 28))
# model.add(Dense(8, activation=keras.activations.sigmoid))
# model.add(Dense(3, activation=keras.activations.sigmoid))
# model.add(Dense(512, activation=keras.activations.sigmoid))
# model.add(Dense(10, activation=keras.activations.sigmoid))
# tf.keras.layers.Dropout(0.2)

model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

# model.add(Dense(8, activation=tf.nn.relu))
# model.add(Dense(3, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# model.compile(optimizer=tf.train.AdamOptimizer(),
#               # loss=keras.losses.sparse_categorical_crossentropy,
#               loss=keras.losses.mse,
#               # metrics=[keras.metrics.sparse_top_k_categorical_accuracy])
#               metrics=[keras.metrics.categorical_accuracy])
#               #   metrics=[keras.metrics.binary_accuracy])

# This is the process I used to train my weights
# model.fit(bin7, count3, epochs=2000)
# myWeights = model.get_weights()
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=2)
# print('myWeights =', myWeights)

model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

myWeights = model.get_weights()
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
print('myWeights =', myWeights)

# test the model and your weights
# model.fit(bin7, count3, epochs=1)
# model.set_weights(myWeights)
# predict3 = model.predict(bin7)
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=1)
# print('prediction =', predict3)

model.fit(x_test, y_test, epochs=1)
# model.fit(x_train, y_train, epochs=1)

model.set_weights(myWeights)
predict3 = model.predict(x_test)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=1)
print('prediction =', predict3)

Examples = {

    # 'MNIST' : [x_train, y_train, model, myWeights ]
    'MNIST': [x_test, y_test, model, myWeights]

    # 'MNIST': [x_train, y_test, model, myWeights]

}

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


