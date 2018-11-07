import tensorflow as tf
from tensorflow import keras
from pprint import pprint
import numpy
from numpy import array, float32

mnist_numbers = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist_numbers.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# train_targets = numpy.zeros([train_labels.size, 10])
#
# for i in range(train_labels.size):
#     train_targets[i][train_labels[i]] = 1.0
#
# test_targets = numpy.zeros([test_labels.size, 10])
#
# for i in range(test_labels.size):
#     test_targets[i][test_labels[i]] = 1.0

train_images = train_images.reshape(*train_images.shape[:1], -1)

two_layer_model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])

two_layer_model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# two_layer_model.fit(train_images, train_labels, epochs=5)

# two_layer_model.save_weights('./weights/my_model_weights.hd5')

two_layer_model.load_weights('../submissions/Chouard/weights/my_model_weights.hd5')

two_layer_weights = two_layer_model.get_weights()

# three_layer_model = keras.Sequential([
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
#
# three_layer_model.compile(optimizer=tf.train.AdamOptimizer(),
#                         loss='sparse_categorical_crossentropy',
#                         metrics=['accuracy'])
#
# three_layer_model.fit(train_images, train_labels, epochs=5)
#
# three_layer_weights = three_layer_model.get_weights()

# pprint(two_layer_weights)
test_labels = test_labels.reshape(*test_labels.shape[:1], -1)
test_images = test_images.reshape(*test_images.shape[:1], -1)

test_loss, test_acc = two_layer_model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

Examples = {
    'images2': [test_images, test_labels, two_layer_model, two_layer_weights],
    # 'images3': [test_images, test_labels, three_layer_model, three_layer_weights],
}