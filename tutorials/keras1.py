import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(64, activation='relu'))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

# # Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# layers.Dense(64, activation=tf.sigmoid)
#
# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))
#
# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')
# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

# model.evaluate(x, y, batch_size=32)
#
# model.evaluate(dataset, steps=30)
#
# model.predict(x, batch_size=32)
#
# model.predict(dataset, steps=30)

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('my_model')

# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')

# Create a trivial model
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('my_model.h5')

