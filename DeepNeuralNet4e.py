import math
import statistics
from utils4e import sigmoid, dotproduct, softmax1D, conv1D, GaussianKernel, element_wise_product, \
    vector_add, random_weights, scalar_vector_product, matrix_multiplication, map_vector, transpose2D
import random
from learning4e import grade_learner, DataSet

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb, cifar10


def cross_entropy_loss(X, Y):
    n = len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def mse_loss(X, Y):
    n = len(X)
    return (1.0/n)*sum((x-y)**2 for x, y in zip(X, Y))


class Node:
    """A node in computational graph, It contains the pointer to all its parents.
    It takes a value which is a tensor"""

    def __init__(self, val=None, parents=[]):
        self.val = val
        self.parents = parents

    def __repr__(self):
        return "<Node {}>".format(self.val)


class NNUnit(Node):
    """Single Unit of a Layer in Neural Network
    inputs: Incoming connections
    weights: Weights to incoming connections
    """

    def __init__(self, weights=None, value=None):
        """value: the computed value of node"""
        super(NNUnit, self).__init__(value)  # input nodes are parent nodes
        self.weights = weights or []


class Layer:
    """Layer based on Computational graph in 19.3.1. A directed graph."""

    def __init__(self, size=3):
        self.nodes = [NNUnit() for _ in range(size)]

    def forward(self, inputs):
        """Define the operation to get the output of this layer"""
        raise NotImplementedError

    def backward(self, nxt):
        """take the information back passed from the next layer
         calculate the information to pass backward"""
        return nxt


# 19.3 Models


class OutputLayer(Layer):
    """Example of a simple 1D softmax output layer in 19.3.2"""
    def __init__(self, size=3):
        super(OutputLayer, self).__init__(size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        res = softmax1D(inputs)
        for node, val in zip(self.nodes, res):
            node.val = val
        return res


class InputLayer(Layer):
    """Simple 1D input layer"""
    def __init__(self, size=3):
        super(InputLayer, self).__init__(size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        for node, inp in zip(self.nodes, inputs):
            node.val = inp
        return inputs


class DenseLayer(Layer):
    """Single 1D dense layer of a neural network
    in_size: input vector size; out_size: output vector size"""

    def __init__(self, in_size=3, out_size=3, activation=None):
        super(DenseLayer, self).__init__(out_size)
        self.out_size = out_size
        self.inputs = None
        self.activation = sigmoid() if not activation else activation
        # initialize weights
        for node in self.nodes:
            node.weights = random_weights(-0.5, 0.5, in_size)

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        for unit in self.nodes:
            val = self.activation.f(dotproduct(unit.weights, inputs))
            unit.val = val
            res.append(val)
        return res


class ConvLayer1D(Layer):
    """Single 1D convolution layer of a neural network
    input channel equals output channel"""

    def __init__(self, size=3, kernel_size=3, kernel_type=None):
        #
        super(ConvLayer1D, self).__init__(size)
        if not kernel_type:
            for node in self.nodes:
                node.weights = GaussianKernel(kernel_size)

    def forward(self, features):
        res = []
        for node, feature in zip(self.nodes, features):
            out = conv1D(feature, node.weights)
            res.append(out)
            node.val = out
        return res


class MaxPoolingLayer1D(Layer):
    """Single 1D max pooling layer in a neural network"""

    def __init__(self, size=3, kernel_size=3):
        super(MaxPoolingLayer1D, self).__init__(size)
        self.kernel_size = kernel_size

    def forward(self, features):
        assert len(self.nodes) == len(features)
        res = []
        for i in range(len(self.nodes)):
            feature = features[i]
            out = [max(feature[i:i+self.kernel_size]) for i in range(len(feature)-self.kernel_size+1)]
            res.append(out)
            self.nodes[i].val = out
        return res

# ____________________________________________________________________
# 19.4 optimization algorithms


def init_examples(examples, idx_i, idx_t, o_units):
    inputs, targets = {}, {}
    # random.shuffle(examples)
    for i, e in enumerate(examples):
        # Input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # One-Hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # Target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


class BatchNormalizationLayer(Layer):

    def __init__(self, inputs, epsilon=0.001):
        super(BatchNormalizationLayer, self).__init__(inputs)
        self.epsilon = epsilon
        self.weights = [0, 0]  # beta and gamma
        self.mu = sum(inputs)/len(inputs)
        self.stderr = statistics.stdev(inputs)

    def forward(self):
        return [(node.val-self.mu)*self.weights[0]/math.sqrt(self.epslong+self.stderr**2)+self.weights[1]
                for node in self.nodes]


def BackPropagation(inputs, targets, theta, net, loss):
    """The back-propagation algorithm for multilayer networks for only one epoch, to calculate gradients of theta
    theta: parameters to be updated """

    assert len(inputs) == len(targets)
    o_units = len(net[-1].nodes)
    n_layers = len(net)
    batch_size = len(inputs)

    gradients = [[[] for _ in layer.nodes] for layer in net]
    total_gradients = [[[0]*len(node.weights) for node in layer.nodes] for layer in net]

    batch_loss = 0

    # iterate over each example in batch
    for e in range(batch_size):
        i_val = inputs[e]
        t_val = targets[e]

        # Forward pass and compute batch loss
        for i in range(1, n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        batch_loss += loss(t_val, layer_out)

        # Initialize delta
        delta = [[] for _ in range(n_layers)]

        previous = [layer_out[i]-t_val[i] for i in range(o_units)]
        h_layers = n_layers - 1
        # Backward pass
        for i in range(h_layers, 0, -1):
            layer = net[i]
            derivative = [layer.activation.derivative(node.val) for node in layer.nodes]
            delta[i] = element_wise_product(previous, derivative)
            # pass to layer i-1 in the next iteration
            previous = matrix_multiplication([delta[i]], theta[i])[0]
            # compute gradient of layer i
            gradients[i] = [scalar_vector_product(d, net[i].inputs) for d in delta[i]]

        # add gradient of current example to batch gradient
        total_gradients = vector_add(total_gradients, gradients)

    return total_gradients, batch_loss


def gradient_descent(dataset, net, loss, epochs=1000, l_rate=0.01,  batch_size=1):
    # init data
    examples = dataset.examples

    for e in range(epochs):
        total_loss = 0
        random.shuffle(examples)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(examples, batch_size):

            inputs, targets = init_examples(batch, dataset.inputs, dataset.target, len(net[-1].nodes))
            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
            # update weights with gradient descent
            weights = vector_add(weights, scalar_vector_product(-l_rate, gs))
            total_loss += batch_loss
            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if (e+1) % 10 == 0:
            print("epoch:{}, total_loss:{}".format(e+1,total_loss))
    return net


def get_batch(examples, batch_size=1):
    """split examples into multiple batches"""
    for i in range(0, len(examples), batch_size):
        yield examples[i: i+batch_size]


def adam_optimizer(dataset, net,  loss, epochs=1000, rho=(0.9, 0.999), delta=1/10**8, l_rate=0.001, batch_size=1):
    examples = dataset.examples
    s = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    r = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    t = 0

    for e in range(epochs):
        total_loss = 0
        random.shuffle(examples)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(examples, batch_size):
            t += 1
            inputs, targets = init_examples(batch, dataset.inputs, dataset.target, len(net[-1].nodes))
            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
            # weights = update(weights, gs, lrate, e+1)
            s = vector_add(scalar_vector_product(rho[0], s),
                           scalar_vector_product((1 - rho[0]), gs))
            r = vector_add(scalar_vector_product(rho[1], r),
                           scalar_vector_product((1 - rho[1]), element_wise_product(gs, gs)))
            s_hat = scalar_vector_product(1 / (1 - rho[0] ** t), s)
            r_hat = scalar_vector_product(1 / (1 - rho[1] ** t), r)
            # rescale r_hat
            r_hat = map_vector(lambda x:1/(math.sqrt(x)+delta), r_hat)
            delta_theta = scalar_vector_product(-l_rate, element_wise_product(s_hat, r_hat))
            weights = vector_add(weights, delta_theta)
            total_loss += batch_loss
            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if (e+1) % 10 == 0:
            print("epoch:{}, total_loss:{}".format(e+1,total_loss))
    return net


def neural_net_learner(dataset, hidden_layer_sizes=[4], learning_rate=0.01, epochs=100, optimizer=gradient_descent):
    """Example of a simple dense multilayer neural network"""
    input_size = len(dataset.inputs)
    output_size = len(dataset.values[dataset.target])

    # initialize the network
    raw_net = [InputLayer(input_size)]

    hidden_input_size = input_size
    for h_size in hidden_layer_sizes:
        raw_net.append(DenseLayer(hidden_input_size, h_size))
        hidden_input_size = h_size
    raw_net.append(DenseLayer(hidden_input_size, output_size))

    learned_net = optimizer(dataset, raw_net, mse_loss, epochs, l_rate=learning_rate)

    def predict(example):
        n_layers = len(learned_net)

        layer_input = example
        layer_out = example
        for i in range(1, n_layers):
            layer_out = learned_net[i].forward(layer_input)
            layer_input = layer_out

        return layer_out.index(max(layer_out))

    return predict


def perceptron_learner(dataset, learning_rate=0.15, epochs=100):
    """Example of a simple dense multilayer neural network"""
    input_size = len(dataset.inputs)
    output_size = len(dataset.values[dataset.target])

    # initialize the network
    raw_net = [DenseLayer(input_size, output_size)]
    learned_net = gradient_descent(dataset, raw_net, mse_loss, epochs, l_rate=learning_rate)

    def predict(example):

        layer_out = learned_net[0].forward(example)
        return layer_out.index(max(layer_out))

    return predict


def simple_rnn_learner(train_data, val_data, epochs=2):

    total_inputs = 5000
    input_length = 500

    # init data
    X_train, y_train = train_data
    X_val, y_val = val_data

    # init model
    model = Sequential()
    model.add(Embedding(total_inputs, 32, input_length=input_length))
    model.add(SimpleRNN(units=128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, verbose=2)

    return model


def keras_dataset_loader(dataset, max_length=500):
    """helper function to load keras datasets"""
    # init dataset
    (X_train, y_train), (X_val, y_val) = dataset
    if max_length>0:
        X_train = sequence.pad_sequences(X_train, maxlen=max_length)
        X_val = sequence.pad_sequences(X_val, maxlen=max_length)
    return (X_train[10:10000], y_train[10:10000]), (X_val, y_val), (X_train[:10], y_train[:10])

