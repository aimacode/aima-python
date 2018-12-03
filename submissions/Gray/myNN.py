# References:
#
# https://www.tensorflow.org/guide/low_level_intro
#

# only needed for python 2.7
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import np_utils
import numpy as np
from numpy import array
from numpy import float32
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
from keras.layers import Activation, Flatten, Dense

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train/255.0
x_test = x_test/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

model.save('train-images-idx3-ubyte.gz')

model.load_weights('train-images-idx3-ubyte.gz')

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)


Examples = {
    #'count3' : [ bin7, count3, model, myWeights ],
    'mnist' : [ x_test, y_test, model, model.get_weights() ]
}


