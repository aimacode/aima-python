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
bin7 = array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
])

'''
Train the network to count to 3
column 0: less than 3
column 1: exactly 3
column 2: more than 3
'''
count3 = array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])


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
# model.fit(bin7, count3, epochs=2000)
# myWeights = model.get_weights()
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=2)
# print('myWeights =', myWeights)

# These are the weights I got, pretty-printed
myWeights = [
    # first layer, 7x8
    array([[ 1.2 , -1.16, -1.97,  2.16,  0.97,  0.86, -1.2 ,  1.12],
       [ 1.21, -1.17, -1.97,  2.16,  0.84,  0.76, -1.19,  1.22],
       [ 1.19, -1.2 , -1.98,  2.15,  0.87,  0.84, -1.19,  1.13],
       [ 1.21, -1.2 , -1.97,  2.15,  0.89,  0.8 , -1.2 ,  1.16],
       [ 1.21, -1.12, -1.97,  2.16,  0.99,  0.8 , -1.21,  1.18],
       [ 1.23, -1.09, -1.98,  2.15,  1.12,  0.81, -1.24,  1.13],
       [ 1.24, -1.11, -1.99,  2.14,  1.  ,  0.77, -1.23,  1.17]],
      dtype=float32),
    # biases for 8 intermediate nodes
    array([-4.57,  3.13,  4.  , -4.44, -1.08, -3.11,  4.39, -4.35],
      dtype=float32),
    # second layer, 8x3
    array([[-2.37, -1.54,  2.82],
       [ 2.57, -0.09, -3.  ],
       [ 3.42, -2.18, -4.26],
       [-3.27,  1.66,  2.1 ],
       [-1.64,  0.12, -0.26],
       [-1.85, -1.73,  2.25],
       [ 2.71,  0.95, -4.85],
       [-2.82, -1.4 ,  2.69]], dtype=float32),
    # biases for 3 output nodes
    array([ 0.21, -0.39, -1.22], dtype=float32)
]


import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Activation
#from keras.callbacks import ModelCheckpoint

#Prepare Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
epochs=100
batch_size=round(x_train.shape[0]/epochs)
classes=10
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

print(x_train.shape)

#regularization divide by number of possible values for RGB
x_train = x_train/255
x_test = x_test/255


#Model
model= keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),filters=32,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))

#model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),activation='relu'))
model.add(keras.layers.Dropout(.5))
#model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(2,2),activation='relu',padding='same'))
model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dropout(.5))
#model.add(keras.layers.Dense(1000,activation='tanh'))
model.add(keras.layers.Dense(10))
model.add(Activation('softmax'))
model.summary()
'''
Basic starting Model
MODEL 4
model= keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),filters=32,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))

#model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),activation='relu'))
model.add(keras.layers.Dropout(.2))
#model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(2,2),activation='relu',padding='same'))
model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
#model.add(keras.layers.Dropout(.5))
#model.add(keras.layers.Dense(1000,activation='tanh'))
model.add(keras.layers.Dense(10))
model.add(Activation('softmax'))

'''



'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          )
'''
#model.save('Convolutional image classifier_Model4Reg.h5')

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=.001),metrics=['categorical_accuracy'])

#model.load_weights('Convolutional image classifier_Model4Reg.h5')






'''Prints out loss and accuracy on a test batch'''

#K=keras.models.Model.test_on_batch(model,x=x_test,y=y_test)
#print('Validation Loss: '+str(K[0])+ '\n'+'Validation Accuracy: '+str(K[1]))



# test the model and your weights
# model.fit(bin7, count3, epochs=1)
# model.set_weights(myWeights)
# predict3 = model.predict(bin7)
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=1)
# print('prediction =', predict3)



Examples = {
    'count3' : [ bin7, count3, model, myWeights ],
    'cifar10': [x_test,y_test,model,model.get_weights()]
}
