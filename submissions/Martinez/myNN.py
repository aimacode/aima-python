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

# a complete input set on 7 bits
# useful for training various sorts of data
# bin7 = array([
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1],
#     [0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 1],
#     [0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0, 1, 1],
#     [0, 0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1],
#     [0, 0, 0, 1, 1, 1, 0],
#     [0, 0, 0, 1, 1, 1, 1],
#     [0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 1, 0],
#     [0, 0, 1, 0, 0, 1, 1],
#     [0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 1, 0, 1, 0, 1],
#     [0, 0, 1, 0, 1, 1, 0],
#     [0, 0, 1, 0, 1, 1, 1],
#     [0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 1],
#     [0, 0, 1, 1, 0, 1, 0],
#     [0, 0, 1, 1, 0, 1, 1],
#     [0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 0, 1],
#     [0, 0, 1, 1, 1, 1, 0],
#     [0, 0, 1, 1, 1, 1, 1],
#     [0, 1, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 1],
#     [0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0, 1, 1],
#     [0, 1, 0, 0, 1, 0, 0],
#     [0, 1, 0, 0, 1, 0, 1],
#     [0, 1, 0, 0, 1, 1, 0],
#     [0, 1, 0, 0, 1, 1, 1],
#     [0, 1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 1],
#     [0, 1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1, 1],
#     [0, 1, 0, 1, 1, 0, 0],
#     [0, 1, 0, 1, 1, 0, 1],
#     [0, 1, 0, 1, 1, 1, 0],
#     [0, 1, 0, 1, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0, 0, 1],
#     [0, 1, 1, 0, 0, 1, 0],
#     [0, 1, 1, 0, 0, 1, 1],
#     [0, 1, 1, 0, 1, 0, 0],
#     [0, 1, 1, 0, 1, 0, 1],
#     [0, 1, 1, 0, 1, 1, 0],
#     [0, 1, 1, 0, 1, 1, 1],
#     [0, 1, 1, 1, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0, 1],
#     [0, 1, 1, 1, 0, 1, 0],
#     [0, 1, 1, 1, 0, 1, 1],
#     [0, 1, 1, 1, 1, 0, 0],
#     [0, 1, 1, 1, 1, 0, 1],
#     [0, 1, 1, 1, 1, 1, 0],
#     [0, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0],
#     [1, 0, 0, 0, 0, 1, 1],
#     [1, 0, 0, 0, 1, 0, 0],
#     [1, 0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1],
#     [1, 0, 0, 1, 0, 0, 0],
#     [1, 0, 0, 1, 0, 0, 1],
#     [1, 0, 0, 1, 0, 1, 0],
#     [1, 0, 0, 1, 0, 1, 1],
#     [1, 0, 0, 1, 1, 0, 0],
#     [1, 0, 0, 1, 1, 0, 1],
#     [1, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 1, 1, 1, 1],
#     [1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 1, 0],
#     [1, 0, 1, 0, 0, 1, 1],
#     [1, 0, 1, 0, 1, 0, 0],
#     [1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 1, 0],
#     [1, 0, 1, 0, 1, 1, 1],
#     [1, 0, 1, 1, 0, 0, 0],
#     [1, 0, 1, 1, 0, 0, 1],
#     [1, 0, 1, 1, 0, 1, 0],
#     [1, 0, 1, 1, 0, 1, 1],
#     [1, 0, 1, 1, 1, 0, 0],
#     [1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 1, 1, 1, 1, 0],
#     [1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 0, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0, 0, 1],
#     [1, 1, 0, 0, 0, 1, 0],
#     [1, 1, 0, 0, 0, 1, 1],
#     [1, 1, 0, 0, 1, 0, 0],
#     [1, 1, 0, 0, 1, 0, 1],
#     [1, 1, 0, 0, 1, 1, 0],
#     [1, 1, 0, 0, 1, 1, 1],
#     [1, 1, 0, 1, 0, 0, 0],
#     [1, 1, 0, 1, 0, 0, 1],
#     [1, 1, 0, 1, 0, 1, 0],
#     [1, 1, 0, 1, 0, 1, 1],
#     [1, 1, 0, 1, 1, 0, 0],
#     [1, 1, 0, 1, 1, 0, 1],
#     [1, 1, 0, 1, 1, 1, 0],
#     [1, 1, 0, 1, 1, 1, 1],
#     [1, 1, 1, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 1],
#     [1, 1, 1, 0, 0, 1, 0],
#     [1, 1, 1, 0, 0, 1, 1],
#     [1, 1, 1, 0, 1, 0, 0],
#     [1, 1, 1, 0, 1, 0, 1],
#     [1, 1, 1, 0, 1, 1, 0],
#     [1, 1, 1, 0, 1, 1, 1],
#     [1, 1, 1, 1, 0, 0, 0],
#     [1, 1, 1, 1, 0, 0, 1],
#     [1, 1, 1, 1, 0, 1, 0],
#     [1, 1, 1, 1, 0, 1, 1],
#     [1, 1, 1, 1, 1, 0, 0],
#     [1, 1, 1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1, 1, 1],
# ])
#
# '''
# Train the network to count to 3
# column 0: less than 3
# column 1: exactly 3
# column 2: more than 3
# '''
# count3 = array([
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 0, 0],
#     [1, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
# ])
## this was done in google colab

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#load whole dataset but only keep the top 5000 words
#50/50 split
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 1000
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
#+64 inputs
model.add(Embedding(top_words, 200, input_length=max_words))
model.add(Flatten())

#86% accuracy
#model.add(Dense(250, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))


# 50% accuracy
# model.add(Dense(15, activation= 'sigmoid'))
# model.add(Dense(25, activation= 'relu'))
# model.add(Dense(1, activation='sigmoid'))


#81% accuracy
# model.add(Dense(10, activation= 'relu'))
# model.add(Dense(15, activation= 'relu'))
# model.add(Dense(25, activation= 'relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1, activation= 'relu'))

#
model.add(Dense(50, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation= 'relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model; lots of data only 2 epochs and a large batch size of 128
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#Accuracy = "Accuracy: %.2f%%" % (scores[1]*100

#print("Accuracy: %.2f%%" % (scores[1]*100))
#print(Accuracy)
Weights = model.get_weights()
Examples = {
    'Imdb' : [X_train, y_train, model, Weights],
}
