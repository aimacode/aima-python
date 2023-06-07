"""Deep learning. (Chapters 20)"""

import random
import statistics

import numpy as np
from keras import Sequential, optimizers
from keras.layers import Embedding, SimpleRNN, Dense
from keras.utils import pad_sequences

from utils4e import (conv1D, gaussian_kernel, element_wise_product, vector_add, random_weights,
                     scalar_vector_product, map_vector, mean_squared_error_loss)


class Node:
    """
    A single unit of a layer in a neural network
    :param weights: weights between parent nodes and current node
    :param value: value of current node
    """

    def __init__(self, weights=None, value=None):
        self.value = value
        self.weights = weights or []


class Layer:
    """
    A layer in a neural network based on a computational graph.
    :param size: number of units in the current layer
    """

    def __init__(self, size):
        self.nodes = np.array([Node() for _ in range(size)])

    def forward(self, inputs):
        """Define the operation to get the output of this layer"""
        raise NotImplementedError


class Activation:

    def function(self, x):
        return NotImplementedError

    def derivative(self, x):
        return NotImplementedError

    def __call__(self, x):
        return self.function(x)


class Sigmoid(Activation):

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, value):
        return value * (1 - value)


class ReLU(Activation):

    def function(self, x):
        return max(0, x)

    def derivative(self, value):
        return 1 if value > 0 else 0


class ELU(Activation):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def function(self, x):
        return x if x > 0 else self.alpha * (np.exp(x) - 1)

    def derivative(self, value):
        return 1 if value > 0 else self.alpha * np.exp(value)


class LeakyReLU(Activation):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def function(self, x):
        return max(x, self.alpha * x)

    def derivative(self, value):
        return 1 if value > 0 else self.alpha


class Tanh(Activation):

    def function(self, x):
        return np.tanh(x)

    def derivative(self, value):
        return 1 - (value ** 2)


class SoftMax(Activation):

    def function(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        return np.ones_like(x)


class SoftPlus(Activation):

    def function(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class Linear(Activation):

    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class InputLayer(Layer):
    """1D input layer. Layer size is the same as input vector size."""

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs):
        """Take each value of the inputs to each unit in the layer."""
        assert len(self.nodes) == len(inputs)
        for node, inp in zip(self.nodes, inputs):
            node.value = inp
        return inputs


class OutputLayer(Layer):
    """1D softmax output layer in 19.3.2."""

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs, activation=SoftMax):
        assert len(self.nodes) == len(inputs)
        res = activation().function(inputs)
        for node, val in zip(self.nodes, res):
            node.value = val
        return res


class DenseLayer(Layer):
    """
    1D dense layer in a neural network.
    :param in_size: (int) input vector size
    :param out_size: (int) output vector size
    :param activation: (Activation object) activation function
    """

    def __init__(self, in_size=3, out_size=3, activation=Sigmoid):
        super().__init__(out_size)
        self.out_size = out_size
        self.inputs = None
        self.activation = activation()
        # initialize weights
        for node in self.nodes:
            node.weights = random_weights(-0.5, 0.5, in_size)

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        # get the output value of each unit
        for unit in self.nodes:
            val = self.activation.function(np.dot(unit.weights, inputs))
            unit.value = val
            res.append(val)
        return res


class ConvLayer1D(Layer):
    """
    1D convolution layer of in neural network.
    :param kernel_size: convolution kernel size
    """

    def __init__(self, size=3, kernel_size=3):
        super().__init__(size)
        # init convolution kernel as gaussian kernel
        for node in self.nodes:
            node.weights = gaussian_kernel(kernel_size)

    def forward(self, features):
        # each node in layer takes a channel in the features
        assert len(self.nodes) == len(features)
        res = []
        # compute the convolution output of each channel, store it in node.val
        for node, feature in zip(self.nodes, features):
            out = conv1D(feature, node.weights)
            res.append(out)
            node.value = out
        return res


class MaxPoolingLayer1D(Layer):
    """
    1D max pooling layer in a neural network.
    :param kernel_size: max pooling area size
    """

    def __init__(self, size=3, kernel_size=3):
        super().__init__(size)
        self.kernel_size = kernel_size
        self.inputs = None

    def forward(self, features):
        assert len(self.nodes) == len(features)
        res = []
        self.inputs = features
        # do max pooling for each channel in features
        for i in range(len(self.nodes)):
            feature = features[i]
            # get the max value in a kernel_size * kernel_size area
            out = [max(feature[i:i + self.kernel_size])
                   for i in range(len(feature) - self.kernel_size + 1)]
            res.append(out)
            self.nodes[i].value = out
        return res


class BatchNormalizationLayer(Layer):
    """Batch normalization layer."""

    def __init__(self, size, eps=0.001):
        super().__init__(size)
        self.eps = eps
        # self.weights = [beta, gamma]
        self.weights = [0, 0]
        self.inputs = None

    def forward(self, inputs):
        # mean value of inputs
        mu = sum(inputs) / len(inputs)
        # standard error of inputs
        stderr = statistics.stdev(inputs)
        self.inputs = inputs
        res = []
        # get normalized value of each input
        for i in range(len(self.nodes)):
            val = [(inputs[i] - mu) * self.weights[0] / np.sqrt(self.eps + stderr ** 2) + self.weights[1]]
            res.append(val)
            self.nodes[i].value = val
        return res


def init_examples(examples, idx_i, idx_t, o_units):
    """Init examples from dataset.examples."""

    inputs, targets = {}, {}
    for i, e in enumerate(examples):
        # input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # one-hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


def stochastic_gradient_descent(dataset, net, loss, epochs=1000, l_rate=0.01, batch_size=1, verbose=False):
    """
    Gradient descent algorithm to update the learnable parameters of a network.
    :return: the updated network
    """
    examples = dataset.examples  # init data

    for e in range(epochs):
        total_loss = 0
        random.shuffle(examples)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(examples, batch_size):
            inputs, targets = init_examples(batch, dataset.inputs, dataset.target, len(net[-1].nodes))
            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
            # update weights with gradient descent
            weights = [x + y for x, y in zip(weights, [np.array(tg) * -l_rate for tg in gs])]
            total_loss += batch_loss

            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i].size != 0:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net


def adam(dataset, net, loss, epochs=1000, rho=(0.9, 0.999), delta=1 / 10 ** 8,
         l_rate=0.001, batch_size=1, verbose=False):
    """
    [Figure 19.6]
    Adam optimizer to update the learnable parameters of a network.
    Required parameters are similar to gradient descent.
    :return the updated network
    """
    examples = dataset.examples

    # init s,r and t
    s = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    r = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    t = 0

    # repeat util converge
    for e in range(epochs):
        # total loss of each epoch
        total_loss = 0
        random.shuffle(examples)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(examples, batch_size):
            t += 1
            inputs, targets = init_examples(batch, dataset.inputs, dataset.target, len(net[-1].nodes))

            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)

            # update s,r,s_hat and r_gat
            s = vector_add(scalar_vector_product(rho[0], s),
                           scalar_vector_product((1 - rho[0]), gs))
            r = vector_add(scalar_vector_product(rho[1], r),
                           scalar_vector_product((1 - rho[1]), element_wise_product(gs, gs)))
            s_hat = scalar_vector_product(1 / (1 - rho[0] ** t), s)
            r_hat = scalar_vector_product(1 / (1 - rho[1] ** t), r)

            # rescale r_hat
            r_hat = map_vector(lambda x: 1 / (np.sqrt(x) + delta), r_hat)

            # delta weights
            delta_theta = scalar_vector_product(-l_rate, element_wise_product(s_hat, r_hat))
            weights = vector_add(weights, delta_theta)
            total_loss += batch_loss

            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net


def BackPropagation(inputs, targets, theta, net, loss):
    """
    The back-propagation algorithm for multilayer networks in only one epoch, to calculate gradients of theta.
    :param inputs: a batch of inputs in an array. Each input is an iterable object
    :param targets: a batch of targets in an array. Each target is an iterable object
    :param theta: parameters to be updated
    :param net: a list of predefined layer objects representing their linear sequence
    :param loss: a predefined loss function taking array of inputs and targets
    :return: gradients of theta, loss of the input batch
    """

    assert len(inputs) == len(targets)
    o_units = len(net[-1].nodes)
    n_layers = len(net)
    batch_size = len(inputs)

    gradients = [[[] for _ in layer.nodes] for layer in net]
    total_gradients = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]

    batch_loss = 0

    # iterate over each example in batch
    for e in range(batch_size):
        i_val = inputs[e]
        t_val = targets[e]

        # forward pass and compute batch loss
        for i in range(1, n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        batch_loss += loss(t_val, layer_out)

        # initialize delta
        delta = [[] for _ in range(n_layers)]

        previous = np.array([layer_out[i] - t_val[i] for i in range(o_units)])
        h_layers = n_layers - 1

        # backward pass
        for i in range(h_layers, 0, -1):
            layer = net[i]
            derivative = np.array([layer.activation.derivative(node.value) for node in layer.nodes])
            delta[i] = previous * derivative
            # pass to layer i-1 in the next iteration
            previous = np.matmul([delta[i]], theta[i])[0]
            # compute gradient of layer i
            gradients[i] = [scalar_vector_product(d, net[i].inputs) for d in delta[i]]

        # add gradient of current example to batch gradient
        total_gradients = vector_add(total_gradients, gradients)

    return total_gradients, batch_loss


def get_batch(examples, batch_size=1):
    """Split examples into multiple batches"""
    for i in range(0, len(examples), batch_size):
        yield examples[i: i + batch_size]


class NeuralNetworkLearner:
    """
    Simple dense multilayer neural network.
    :param hidden_layer_sizes: size of hidden layers in the form of a list
    """

    def __init__(self, dataset, hidden_layer_sizes, l_rate=0.01, epochs=1000, batch_size=10,
                 optimizer=stochastic_gradient_descent, loss=mean_squared_error_loss, verbose=False, plot=False):
        self.dataset = dataset
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.plot = plot

        input_size = len(dataset.inputs)
        output_size = len(dataset.values[dataset.target])

        # initialize the network
        raw_net = [InputLayer(input_size)]
        # add hidden layers
        hidden_input_size = input_size
        for h_size in hidden_layer_sizes:
            raw_net.append(DenseLayer(hidden_input_size, h_size))
            hidden_input_size = h_size
        raw_net.append(DenseLayer(hidden_input_size, output_size))
        self.raw_net = raw_net

    def fit(self, X, y):
        self.learned_net = self.optimizer(self.dataset, self.raw_net, loss=self.loss, epochs=self.epochs,
                                          l_rate=self.l_rate, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, example):
        n_layers = len(self.learned_net)

        layer_input = example
        layer_out = example

        # get the output of each layer by forward passing
        for i in range(1, n_layers):
            layer_out = self.learned_net[i].forward(np.array(layer_input).reshape((-1, 1)))
            layer_input = layer_out

        return layer_out.index(max(layer_out))


class PerceptronLearner:
    """
    Simple perceptron neural network.
    """

    def __init__(self, dataset, l_rate=0.01, epochs=1000, batch_size=10, optimizer=stochastic_gradient_descent,
                 loss=mean_squared_error_loss, verbose=False, plot=False):
        self.dataset = dataset
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.plot = plot

        input_size = len(dataset.inputs)
        output_size = len(dataset.values[dataset.target])

        # initialize the network, add dense layer
        self.raw_net = [InputLayer(input_size), DenseLayer(input_size, output_size)]

    def fit(self, X, y):
        self.learned_net = self.optimizer(self.dataset, self.raw_net, loss=self.loss, epochs=self.epochs,
                                          l_rate=self.l_rate, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, example):
        layer_out = self.learned_net[1].forward(np.array(example).reshape((-1, 1)))
        return layer_out.index(max(layer_out))


def keras_dataset_loader(dataset, max_length=500):
    """
    Helper function to load keras datasets.
    :param dataset: keras data set type
    :param max_length: max length of each input sequence
    """
    # init dataset
    (X_train, y_train), (X_val, y_val) = dataset
    if max_length > 0:
        X_train = pad_sequences(X_train, maxlen=max_length)
        X_val = pad_sequences(X_val, maxlen=max_length)
    return (X_train[10:], y_train[10:]), (X_val, y_val), (X_train[:10], y_train[:10])


def SimpleRNNLearner(train_data, val_data, epochs=2, verbose=False):
    """
    RNN example for text sentimental analysis.
    :param train_data: a tuple of (training data, targets)
            Training data: ndarray taking training examples, while each example is coded by embedding
            Targets: ndarray taking targets of each example. Each target is mapped to an integer
    :param val_data: a tuple of (validation data, targets)
    :param epochs: number of epochs
    :param verbose: verbosity mode
    :return: a keras model
    """

    total_inputs = 5000
    input_length = 500

    # init data
    X_train, y_train = train_data
    X_val, y_val = val_data

    # init a the sequential network (embedding layer, rnn layer, dense layer)
    model = Sequential()
    model.add(Embedding(total_inputs, 32, input_length=input_length))
    model.add(SimpleRNN(units=128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, verbose=verbose)

    return model


def AutoencoderLearner(inputs, encoding_size, epochs=200, verbose=False):
    """
    Simple example of linear auto encoder learning producing the input itself.
    :param inputs: a batch of input data in np.ndarray type
    :param encoding_size: int, the size of encoding layer
    :param epochs: number of epochs
    :param verbose: verbosity mode
    :return: a keras model
    """

    # init data
    input_size = len(inputs[0])

    # init model
    model = Sequential()
    model.add(Dense(encoding_size, input_dim=input_size, activation='relu', kernel_initializer='random_uniform',
                    bias_initializer='ones'))
    model.add(Dense(input_size, activation='relu', kernel_initializer='random_uniform', bias_initializer='ones'))

    # update model with sgd
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # train the model
    model.fit(inputs, inputs, epochs=epochs, batch_size=10, verbose=verbose)

    return model
