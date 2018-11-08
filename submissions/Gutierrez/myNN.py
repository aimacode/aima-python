from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import fashion_mnist
import tensorflow as tf

data = fashion_mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

# normilization here helps the model while learning, compressing the values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#CANNOT GET FLATTEN TO BE RECOGNISED AS A VALID TENSOR INOUT .    had to import *face palm*
model = Sequential()
model.add(Flatten()) #faltten flattens the imput without effecting batch size
model.add(Dense(100, activation=tf.nn.relu))
model.add(Dense(100, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))  # softmax for probability sake
#sgd =.86, Adagrad =.90, RMSprop =.90, Adadelta =.91, Adam= .91, Adamax= .90, Nadam .91
model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

model.summary()
#
# Dataset of 60,000 28x28 grayscale images of 10 fashion categories

# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
