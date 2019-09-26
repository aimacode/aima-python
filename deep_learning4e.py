import math
import statistics

from utils4e import sigmoid, dotproduct, softmax1D, conv1D, gaussian_kernel_2d, GaussianKernel, element_wise_product, \
    vector_add, random_weights, scalar_vector_product, matrix_multiplication, map_vector
import random

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# DEEP NEURAL NETWORKS. (Chapter 19)
# ________________________________________________
# 19.2 Common Loss Functions


def cross_entropy_loss(X, Y):
    """Example of cross entropy loss. X and Y are 1D iterable objects"""
    n = len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def mse_loss(X, Y):
    """Example of min square loss. X and Y are 1D iterable objects"""
    n = len(X)
    return (1.0/n)*sum((x-y)**2 for x, y in zip(X, Y))

# ________________________________________________
# 19.3 Models
# 19.3.1 Computational Graphs and Layers


class Node:
    """
    A node in computational graph, It contains the pointer to all its parents.
    :param val: value of current node.
    :param parents: a container of all parents of current node.
    """

    def __init__(self, val=None, parents=[]):
        self.val = val
        self.parents = parents

    def __repr__(self):
        return "<Node {}>".format(self.val)


class NNUnit(Node):
    """
    A single unit of a Layer in a Neural Network
    :param weights: weights between parent nodes and current node
    :param value: value of current node
    """

    def __init__(self, weights=None, value=None):
        super(NNUnit, self).__init__(value)
        self.weights = weights or []


class Layer:
    """
    A layer in a neural network based on computational graph.
    :param size: number of units in the current layer
    """

    def __init__(self, size=3):
        self.nodes = [NNUnit() for _ in range(size)]

    def forward(self, inputs):
        """Define the operation to get the output of this layer"""
        raise NotImplementedError


# 19.3.2 Output Layers


class OutputLayer(Layer):
    """Example of a 1D softmax output layer in 19.3.2"""
    def __init__(self, size=3):
        super(OutputLayer, self).__init__(size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        res = softmax1D(inputs)
        for node, val in zip(self.nodes, res):
            node.val = val
        return res


class InputLayer(Layer):
    """Example of a 1D input layer. Layer size is the same as input vector size."""
    def __init__(self, size=3):
        super(InputLayer, self).__init__(size)

    def forward(self, inputs):
        """Take each value of the inputs to each unit in the layer."""
        assert len(self.nodes) == len(inputs)
        for node, inp in zip(self.nodes, inputs):
            node.val = inp
        return inputs

# 19.3.3 Hidden Layers


class DenseLayer(Layer):
    """
    1D dense layer in a neural network.
    :param in_size: input vector size, int.
    :param out_size: output vector size, int.
    :param activation: activation function, Activation object.
    """

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
        # get the output value of each unit
        for unit in self.nodes:
            val = self.activation.f(dotproduct(unit.weights, inputs))
            unit.val = val
            res.append(val)
        return res

# 19.3.4 Convolutional networks


class ConvLayer1D(Layer):
    """
    1D convolution layer of in neural network.
    :param kernel_size: convolution kernel size
    """

    def __init__(self, size=3, kernel_size=3):
        super(ConvLayer1D, self).__init__(size)
        # init convolution kernel as gaussian kernel
        for node in self.nodes:
            node.weights = GaussianKernel(kernel_size)

    def forward(self, features):
        # Each node in layer takes a channel in the features.
        assert len(self.nodes) == len(features)
        res = []
        # compute the convolution output of each channel, store it in node.val.
        for node, feature in zip(self.nodes, features):
            out = conv1D(feature, node.weights)
            res.append(out)
            node.val = out
        return res

# 19.3.5 Pooling and Downsampling


class MaxPoolingLayer1D(Layer):
    """1D max pooling layer in a neural network.
    :param kernel_size: max pooling area size"""

    def __init__(self, size=3, kernel_size=3):
        super(MaxPoolingLayer1D, self).__init__(size)
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
            out = [max(feature[i:i+self.kernel_size]) for i in range(len(feature)-self.kernel_size+1)]
            res.append(out)
            self.nodes[i].val = out
        return res

# ____________________________________________________________________
# 19.4 optimization algorithms


def init_examples(examples, idx_i, idx_t, o_units):
    """Init examples from dataset.examples."""

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

# 19.4.1 Stochastic gradient descent


def gradient_descent(dataset, net, loss, epochs=1000, l_rate=0.01,  batch_size=1, verbose=None):
    """
    gradient descent algorithm to update the learnable parameters of a network.
    :return: the updated network.
    """
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

        if verbose and (e+1) % verbose == 0:
            print("epoch:{}, total_loss:{}".format(e+1,total_loss))
    return net


# 19.4.2 Other gradient-based optimization algorithms


def adam_optimizer(dataset, net,  loss, epochs=1000, rho=(0.9, 0.999), delta=1/10**8, l_rate=0.001, batch_size=1, verbose=None):
    """
    Adam optimizer in Figure 19.6 to update the learnable parameters of a network.
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
            r_hat = map_vector(lambda x: 1/(math.sqrt(x)+delta), r_hat)
            # delta weights
            delta_theta = scalar_vector_product(-l_rate, element_wise_product(s_hat, r_hat))
            weights = vector_add(weights, delta_theta)
            total_loss += batch_loss
            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose and (e+1) % verbose == 0:
            print("epoch:{}, total_loss:{}".format(e+1,total_loss))
    return net

# 19.4.3 Back-propagation


def BackPropagation(inputs, targets, theta, net, loss):
    """
    The back-propagation algorithm for multilayer networks in only one epoch, to calculate gradients of theta
    :param inputs: A batch of inputs in an array. Each input is an iterable object.
    :param targets: A batch of targets in an array. Each target is an iterable object.
    :param theta: parameters to be updated.
    :param net: a list of predefined layer objects representing their linear sequence.
    :param loss: a predefined loss function taking array of inputs and targets.
    :return: gradients of theta, loss of the input batch.
    """

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

# 19.4.5 Batch normalization


class BatchNormalizationLayer(Layer):
    """Example of a batch normalization layer."""
    def __init__(self, size, epsilon=0.001):
        super(BatchNormalizationLayer, self).__init__(size)
        self.epsilon = epsilon
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
            val = [(inputs[i] - mu)*self.weights[0]/math.sqrt(self.epsilon + stderr**2)+self.weights[1]]
            res.append(val)
            self.nodes[i].val = val
        return res


def get_batch(examples, batch_size=1):
    """split examples into multiple batches"""
    for i in range(0, len(examples), batch_size):
        yield examples[i: i+batch_size]

# example of NNs


def neural_net_learner(dataset, hidden_layer_sizes=[4], learning_rate=0.01, epochs=100, optimizer=gradient_descent, batch_size=1, verbose=None):
    """Example of a simple dense multilayer neural network.
    :param hidden_layer_sizes: size of hidden layers in the form of a list"""

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

    # update parameters of the network
    learned_net = optimizer(dataset, raw_net, mse_loss, epochs, l_rate=learning_rate, batch_size=batch_size, verbose=verbose)

    def predict(example):
        n_layers = len(learned_net)

        layer_input = example
        layer_out = example

        # get the output of each layer by forward passing
        for i in range(1, n_layers):
            layer_out = learned_net[i].forward(layer_input)
            layer_input = layer_out

        return layer_out.index(max(layer_out))

    return predict


def perceptron_learner(dataset, learning_rate=0.01, epochs=100, verbose=None):
    """
    Example of a simple perceptron neural network.
    """
    input_size = len(dataset.inputs)
    output_size = len(dataset.values[dataset.target])

    # initialize the network, add dense layer
    raw_net = [InputLayer(input_size), DenseLayer(input_size, output_size)]
    # update the network
    learned_net = gradient_descent(dataset, raw_net, mse_loss, epochs, l_rate=learning_rate, verbose=verbose)

    def predict(example):

        layer_out = learned_net[1].forward(example)
        return layer_out.index(max(layer_out))

    return predict

# ____________________________________________________________________
# 19.6 Recurrent neural networks


def simple_rnn_learner(train_data, val_data, epochs=2):
    """
    rnn example for text sentimental analysis
    :param train_data: a tuple of (training data, targets)
            Training data: ndarray taking training examples, while each example is coded by embedding
            Targets: ndarry taking targets of each example. Each target is mapped to an integer.
    :param val_data: a tuple of (validation data, targets)
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, verbose=2)

    return model


def keras_dataset_loader(dataset, max_length=500):
    """
    helper function to load keras datasets
    :param dataset: keras data set type
    :param max_length: max length of each input sequence
    """
    # init dataset
    (X_train, y_train), (X_val, y_val) = dataset
    if max_length > 0:
        X_train = sequence.pad_sequences(X_train, maxlen=max_length)
        X_val = sequence.pad_sequences(X_val, maxlen=max_length)
    return (X_train[10:], y_train[10:]), (X_val, y_val), (X_train[:10], y_train[:10])


def auto_encoder_learner(inputs, encoding_size, epochs=200):
    """simple example of linear auto encoder learning producing the input itself.
    :param inputs: a batch of input data in np.ndarray type
    :param encoding_size: int, the size of encoding layer"""

    # init data
    input_size = len(inputs[0])

    # init model
    model = Sequential()
    model.add(Dense(encoding_size, input_dim=input_size, activation='relu', kernel_initializer='random_uniform',bias_initializer='ones'))
    model.add(Dense(input_size, activation='relu', kernel_initializer='random_uniform', bias_initializer='ones'))
    # update model with sgd
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # train the model
    model.fit(inputs, inputs, epochs=epochs, batch_size=10, verbose=2)

    return model
