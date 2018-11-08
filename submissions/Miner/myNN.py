import numpy as np
import tensorflow as tf

imageData = tf.keras.datasets.cifar10


def normalize(data):
    data = data.astype('float32')
    # Pixel values between 0 and 255
    return data / 255.0


(x_train, y_train), (x_test, y_test) = imageData.load_data()

x_train = normalize(x_train)
x_test = normalize(x_test)

y_train = normalize(y_train)
y_test = normalize(y_test)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

epochs = 1
batch_size = 32

# input shape
x = x_train.shape[1]
y = x_train.shape[2]
z = x_train.shape[3]

# 1st and 2nd try sgd, seems to work pretty well
sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01/epochs, nesterov=False)

# 3rd try sgd, increasing learning rate
# sgd = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.01/epochs, nesterov=False)

model = tf.keras.models.Sequential()


# Begin the learning / 1st try. Got too 100%?
# model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(x, y, z), padding='same', activation='relu'))
# model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(x, y, z)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 2nd try, with Dropout layers to prevent over-fitting, .995 accuracy
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(x, y, z), padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(x, y, z)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 3rd try, removing Conv2d layer and dropouts....still 98%???
# model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(x, y, z), padding='same', activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax', input_shape=(x, y, z)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.summary()

weights = model.get_weights()

# image_results = model.fit(
#     x_train, y_train,
#     validation_data=(x_test, y_test),
#     batch_size=batch_size,
#     epochs=epochs
# )

# print("Test-Accuracy:", image_results.history)


# ============================================================================================================
# IMDB Binary Categorical NN
# ============================================================================================================
imdbData = tf.keras.datasets.imdb

(training_data, training_targets), (testing_data, testing_targets) = imdbData.load_data(num_words=10000)
imdb_data = np.concatenate((training_data, testing_data), axis=0)
imdb_targets = np.concatenate((training_targets, testing_targets), axis=0)


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


imdb_data = vectorize(imdb_data)
imdb_targets = np.array(imdb_targets).astype("float32")

test_x = imdb_data[:10000]
test_y = imdb_targets[:10000]
train_x = imdb_data[10000:]
train_y = imdb_targets[10000:]

imdb_model = tf.keras.models.Sequential()

# 88% acc
imdb_model.add(tf.keras.layers.Dense(50, activation="relu", input_shape=(10000, )))
imdb_model.add(tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
imdb_model.add(tf.keras.layers.Dense(50, activation="relu"))
imdb_model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))
imdb_model.add(tf.keras.layers.Dense(50, activation="relu"))
imdb_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
imdb_model.summary()

imdb_weights = imdb_model.get_weights()

imdb_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# imdb_results = imdb_model.fit(
#     train_x, train_y,
#     epochs=10,
#     batch_size=500,
#     validation_data=(test_x, test_y)
# )

# print("Test-Accuracy:", np.mean(imdb_results.history["val_acc"]))

Examples = {
    'image_10_category': [x_train, y_train, model, weights],
    'imdb_binary': [train_x, train_y, imdb_model, imdb_weights]
}






