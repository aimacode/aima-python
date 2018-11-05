import tensorflow as tf
from tensorflow import keras

mnist_numbers = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist_numbers.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28))
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

