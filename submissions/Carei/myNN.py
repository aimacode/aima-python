#Braden Carei

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
#Vectorizing the data
def vectorization(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

data = np.concatenate((training_data, testing_data), axis=0)

targets = np.concatenate((training_targets, testing_targets), axis=0)

data = vectorization(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
model = models.Sequential()
#Layer for the inputs
model.add(layers.Dense(10, activation="relu", input_shape=(10000,)))
# Layers that are hidden
model.add(layers.Dropout(0.3, noise_shape=None, seed=2))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=2))
#model.add(layers.Dense(10, activation="sigmoid"))
# Layer for the outputs
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
#Model comp
model.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
results = model.fit(
    train_x, train_y,
    batch_size=1000,
    epochs=2,

    validation_data=(test_x, test_y)
)

weights = model.get_weights()

Examples = {
    'imdb dataset': [train_x, train_y, model, weights],

}
