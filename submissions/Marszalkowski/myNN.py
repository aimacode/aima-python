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

'''
Train the network to count to 3
column 0: less than 3
column 1: exactly 3
column 2: more than 3
'''


# this takes a looong time to index, and
# python may crash several times before indexing is complete
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


BHDataSet = tf.keras.datasets.boston_housing

(x_train, y_train), (x_test, y_test) = BHDataSet.load_data()

model = Sequential()
model.add(Dense(8,
                activation=keras.activations.sigmoid,
                ))
model.add(Dense(3,
                activation=keras.activations.sigmoid,
                ))

model.compile(
              optimizer=tf.train.AdamOptimizer(0.001),
              #loss=keras.losses.categorical_crossentropy,
              loss=keras.losses.mse,
              metrics=[keras.metrics.binary_accuracy]
              )

# This is the process I used to train my weights
model.fit(x_train, y_train, epochs=10000)
myWeights = model.get_weights()
np.set_printoptions(suppress=True)
print('myWeights =', myWeights)

# These are the weights I got, pretty-printed
myWeights = [array([[-0.45768797, -0.48737806, -0.41575837,  0.47084966,  0.05686022,
        -0.34761578, -0.47915316,  0.39975634],
       [ 0.08902616, -0.1211257 , -0.43021467,  0.34700742,  0.50143033,
         0.31071806,  0.32518455,  0.31970268],
       [-0.40211067,  0.41535425, -0.44993258,  0.815483  , -0.08692687,
         0.32271338,  0.01617404, -0.16234745],
       [ 0.35127828,  0.42093557, -0.3836938 , -0.1167023 , -0.4210458 ,
         0.05773365, -0.2381717 ,  0.5473    ],
       [-0.38624993, -0.20108147,  0.44598192,  0.35809368,  0.2480351 ,
         0.4008633 ,  0.3169134 , -0.1537862 ],
       [ 0.5812723 , -0.06523906,  0.4787733 ,  0.25097987, -0.08986334,
         0.00209051,  0.41598257, -0.25335726],
       [-0.17026274, -0.5203231 , -0.48432636,  0.17443296, -0.08631498,
        -0.13359112, -0.21258494,  0.01633674],
       [-0.3048431 , -0.09276906,  0.16630155, -0.3540066 ,  0.30097395,
         0.33429587, -0.20458557,  0.03724077],
       [-0.01826641, -0.49944997, -0.01519126,  0.6908332 , -0.19540352,
         0.27978605,  0.2996249 , -0.25350937],
       [ 0.30384398,  0.30286002, -0.40574783, -0.08019371,  0.67109233,
         0.12772846,  0.2935726 ,  0.186077  ],
       [ 0.06393388, -0.13777265,  0.11303717,  0.5744661 ,  0.00238386,
         0.02896321,  0.55426425,  0.00886714],
       [-0.06994793, -0.48804945,  0.05082744,  0.3430536 , -0.27066156,
         0.29923737, -0.10877872, -0.06018113],
       [ 0.09404217,  0.02887591, -0.47390878,  0.42696753, -0.25136632,
         0.30900156,  0.42304477,  0.21060923]], dtype=float32), array([ 0.08142484, -0.05160909, -0.        ,  0.20132683,  0.20649944,
        0.        ,  0.02814963,  0.08799311], dtype=float32), array([[ 2.210736  ,  2.3580213 ,  2.513081  ],
       [ 0.34080938,  0.37752825,  0.81581   ],
       [ 0.18377268, -0.5558785 , -0.6193082 ],
       [ 2.5363495 ,  2.1223004 ,  2.2488348 ],
       [ 2.0039122 ,  1.7048414 ,  2.23254   ],
       [ 1.921546  ,  1.7637306 ,  1.429512  ],
       [ 1.7100188 ,  1.6819048 ,  1.8326395 ],
       [ 4.7416267 ,  5.38621   ,  5.102161  ]], dtype=float32), array([1.548578 , 1.626479 , 1.3220694], dtype=float32)]
# test the model and your weights
model.fit(x_train, y_train, epochs=1)
model.set_weights(myWeights)
predict3 = model.predict(x_train)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=1)
print('prediction =', predict3)

Examples = {
    'BH_Example': [x_train, y_train, model, myWeights],
}
