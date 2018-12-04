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

from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


print (x_train.shape)
print (x_train.dtype)

print (y_train.shape)
print (y_train.dtype)

np.reshape(y_train, (404, -1))

print(y_train.shape)






# this takes a looong time to index, and
# python may crash several times before indexing is complete
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


model = Sequential()


model.add(Dense(8,
                activation=keras.activations.sigmoid,
                ))
model.add(Dense(3,
                activation=keras.activations.sigmoid,
                ))

model.compile(
              optimizer=tf.train.AdamOptimizer(0.001),
              # loss=keras.losses.categorical_crossentropy,
              loss=keras.losses.mse,
              metrics=[keras.metrics.binary_accuracy]
              )

# This is the process I used to train my weights
model.fit(x_train, y_train, epochs=2000)
myWeights = model.get_weights()
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
print('myWeights =', myWeights)

# These are the weights I got, pretty-printed
# myWeights = [
# #     # first layer, 7x8
#     array([[ 1.2 , -1.16, -1.97,  2.16,  0.97,  0.86, -1.2 ,  1.12],
#        [ 1.21, -1.17, -1.97,  2.16,  0.84,  0.76, -1.19,  1.22],
#        [ 1.19, -1.2 , -1.98,  2.15,  0.87,  0.84, -1.19,  1.13],
#        [ 1.21, -1.2 , -1.97,  2.15,  0.89,  0.8 , -1.2 ,  1.16],
#        [ 1.21, -1.12, -1.97,  2.16,  0.99,  0.8 , -1.21,  1.18],
#        [ 1.23, -1.09, -1.98,  2.15,  1.12,  0.81, -1.24,  1.13],
#        [ 1.24, -1.11, -1.99,  2.14,  1.  ,  0.77, -1.23,  1.17]],
#       dtype=float32),
#     # biases for 8 intermediate nodes
#     array([-4.57,  3.13,  4.  , -4.44, -1.08, -3.11,  4.39, -4.35],
#       dtype=float32),
#     # second layer, 8x3
#     array([[-2.37, -1.54,  2.82],
#        [ 2.57, -0.09, -3.  ],
#        [ 3.42, -2.18, -4.26],
#        [-3.27,  1.66,  2.1 ],
#        [-1.64,  0.12, -0.26],
#        [-1.85, -1.73,  2.25],
#        [ 2.71,  0.95, -4.85],
#        [-2.82, -1.4 ,  2.69]], dtype=float32),
#     # biases for 3 output nodes
#     array([ 0.21, -0.39, -1.22], dtype=float32)
# ]

# test the model and your weights
# model.fit(bin7, count3, epochs=1)
# model.set_weights(myWeights)
# predict3 = model.predict(bin7)
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=1)
# print('prediction =', predict3)






Examples = {
    # 'count3' : [ bin7, count3, model, myWeights ],
    'bostonHousing' : [x_train, y_train, model, myWeights]

}
