import math
import statistics
from utils4e import sigmoid, dotproduct, softmax1D, conv1D, GaussianKernel, element_wise_product, \
    vector_add, random_weights, scalar_vector_product, matrix_multiplication, transpose2D, leaky_relu
from learning4e import DataSet
import numpy as np


def cross_entropy_loss(X, Y):
    n=len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def mse_loss(X, Y):
    n = len(X)
    return (1.0/2)*sum((x-y)**2 for x,y in zip(X, Y))


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

    def __init__(self, weights=None, activation=sigmoid(), value=None):
        """value: the computed value of node"""
        super(NNUnit, self).__init__(value)  # input nodes are parent nodes
        self.weights = weights or []
        self.activation = activation


class Layer:
    """Layer based on Computational graph in 19.3.1. A directed graph."""

    def __init__(self, size=3):
        self.nodes = [NNUnit() for _ in range(size)]

    def forward(self, inputs):
        """Define the operation to get the output of this layer"""
        raise NotImplementedError


# 19.3 Models


class OutputLayer(Layer):
    """
    Example of a simple 1D softmax output layer in 19.3.2
    """
    def __init__(self, size=3):
        super(OutputLayer, self).__init__(size)

    def forward(self, inputs):
        if self.size != len(inputs):
            raise ValueError
        outvalues = softmax1D(inputs)
        for node,val in zip(self.nodes,outvalues):
            node.val = val
        return outvalues


class InputLayer(Layer):
    def __init__(self, size=3):
        super(InputLayer, self).__init__(size)

    def forward(self, inputs):
        for node, input in zip(self.nodes, inputs):
            node.val = input
        return inputs


class DenseLayer(Layer):
    """Single dense layer of a neural network
    inputs: NN units contained by the layer"""

    def __init__(self, in_size=3, out_size=3):
        super(DenseLayer, self).__init__(out_size)
        self.out_size = out_size
        # initialize weights
        for node in self.nodes:
            node.weights = random_weights(-0.5, 0.5, in_size)
            # node.weights = [1] * in_size

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        for unit in self.nodes:
            # print(sum(element_wise_product(unit.weights, [i] * len(unit.weights))))
            val = unit.activation.f(sum(element_wise_product(unit.weights, inputs)))
            unit.val = val
            res.append(val)
        return res

    def backward(self, nx_layer, delta):
        h_units = len(self.nodes)
        w = [[node.weights[k] for node in nx_layer.nodes] for k in range(h_units)]
        return [self.nodes[j].activation.derivative(self.nodes[j].val) * dotproduct(w[j], delta)
                for j in range(h_units)]


class ConvLayer1D(Layer):
    """Single conv layer of a neural network"""

    def __init__(self,size=3, kernel_size=3, kernel_type=None):
        # size = output size
        super(ConvLayer1D, self).__init__(size)
        if not kernel_type:
            for node in self.nodes:
                node.weights = GaussianKernel(kernel_size)

    def forward(self, features):
        # padding?
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
        # padding?
        res = []
        for i in range(len(self.nodes)):
            feature = features[i]
            out = [max(feature[i:i+self.kernel_size]) for i in range(len(feature)-self.kernel_size+1)]
            res.append(out)
            self.nodes[i].val = out
        return [res]


class ResidualLayer1D(Layer):
    pass

# ____________________________________________________________________
# 19.4 optimization algorithms


def SGD(theta, gradients, net, lrate=0.5):
    """
    # call backprop to calculate the gradients
    """
    for i in range(len(net)):
        if gradients[i]:
            for j in range(len(gradients[i])):
                theta[i][j] = list(vector_add(theta[i][j], list(
                    scalar_vector_product(-lrate, gradients[i][j]))))
    return 0




# def AaamOptimizer(model, loss, dataset, theta, rho=(0.9, 0.999), lrate=0.001, epoch=1000, delta=10*math.exp(-8)):
#     s = r = [0 for _ in range(theta)]
#     t = 0
#     for e in epoch:
#         gradients = BackPropagationLearner(dataset, model, loss)
#         t += 1
#         s = rho[0] * s + (1-rho[1]) * gradients
#         r = rho[0] * s + (1-rho[1]) * elem_wise(gradients, gradients)
#         s_hat = s/(1-rho[0]*math.exp(t))
#         r_hat = r/(1-rho[1]*math.exp(t))
#         delta_theta = [-s_x(math.sqrt(r_x)+delta) for s_x, r_x in zip(s_hat, r_hat)]
#         return [t + d for t, d in zip(theta, delta_theta)]


def BackPropagationLearner(dataset, net, loss, activation=leaky_relu):
    """The back-propagation algorithm for multilayer networks for only one epoch"""
    # Initialise weights outside of backprop

    examples = dataset.examples  # dataset should be a dataset batch
    '''
    As of now dataset.target gives an int instead of list,
    Changing dataset class will have effect on all the learners.
    Will be taken care of later.
    '''
    o_nodes = net[-1].nodes
    o_units = len(o_nodes)
    idx_t = dataset.target
    idx_i = dataset.inputs
    n_layers = len(net)

    inputs, targets = init_examples(examples, idx_i, idx_t, o_units)
    gradients = [[] for _ in range(n_layers)]
    weights = [[node.weights for node in layer.nodes] for layer in net]
    # Iterate over each example

    l = 0
    for e in range(len(examples)):
        i_val = inputs[e]
        t_val = targets[e]

        # Forward pass
        for i in range(n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        # print(layer_out)

        # Initialize delta
        delta = [[] for _ in range(n_layers)]
        l += loss(t_val, layer_out)
        # compute loss
        err = [layer_out[i]-t_val[i] for i in range(o_units)]
        delta[-1] = [activation().derivative(o_nodes[i].val) * err[i] for i in range(o_units)]
        gradients[-1] = [scalar_vector_product(d/len(examples), net[-1].inputs) for d in delta[-1]]

        # Backward pass
        h_layers = n_layers - 2
        for i in range(h_layers, 0, -1):
            nx_layer = net[i+1]
            nx_w = [node.weights for node in nx_layer.nodes]
            layer = net[i]
            # weights from each ith layer node to each i + 1th layer node
            derivates = [activation().derivative(node.val) for node in layer.nodes]
            delta[i] = element_wise_product(matrix_multiplication([delta[i+1]], nx_w)[0], derivates)
            gradients[i] = [scalar_vector_product(d/len(examples), net[i].inputs) for d in delta[i]]

        SGD(weights, gradients, net)

    for i in range(len(net)):
        if gradients[i]:
            for j in range(len(gradients[i])):
                net[i].nodes[j].weights = weights[i][j]
    print(l)
    print(layer_out)

    return net


def init_examples(examples, idx_i, idx_t, o_units):
    inputs, targets = {}, {}

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

    def __init__(self, inputs, epslong=0.001):
        super(BatchNormalizationLayer, self).__init__(inputs)
        self.epslong = epslong
        self.weights = [0, 0]  # beta and gamma
        self.mu = sum(inputs)/len(inputs)
        self.stderr = statistics.stdev(inputs)

    def forward(self):
        return [(node.value-self.mu)*self.weights[0]/math.sqrt(self.epslong+self.stderr**2)+self.weights[1]
                for node in self.nodes]


def BackPropagation(dataset, weights, net, loss, activation=leaky_relu):
    """The back-propagation algorithm for multilayer networks for only one epoch"""
    # Initialise weights outside of backprop

    examples = dataset.examples  # dataset should be a dataset batch
    '''
    As of now dataset.target gives an int instead of list,
    Changing dataset class will have effect on all the learners.
    Will be taken care of later.
    '''
    o_nodes = net[-1].nodes
    o_units = len(o_nodes)
    idx_t = dataset.target
    idx_i = dataset.inputs
    n_layers = len(net)

    inputs, targets = init_examples(examples, idx_i, idx_t, o_units)
    gradients = [[] for _ in range(n_layers)]
    # Iterate over each example

    l = 0
    for e in range(len(examples)):
        i_val = inputs[e]
        t_val = targets[e]

        # Forward pass
        for i in range(n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        # print(layer_out)

        # Initialize delta
        delta = [[] for _ in range(n_layers)]
        l += loss(t_val, layer_out)
        # compute loss
        err = [layer_out[i]-t_val[i] for i in range(o_units)]
        delta[-1] = [activation().derivative(o_nodes[i].val) * err[i] for i in range(o_units)]
        gradients[-1] = [scalar_vector_product(d/len(examples), net[-1].inputs) for d in delta[-1]]

        # Backward pass
        h_layers = n_layers - 2
        for i in range(h_layers, 0, -1):
            nx_layer = net[i+1]
            nx_w = [node.weights for node in nx_layer.nodes]
            layer = net[i]
            # weights from each ith layer node to each i + 1th layer node
            derivates = [activation().derivative(node.val) for node in layer.nodes]
            delta[i] = element_wise_product(matrix_multiplication([delta[i+1]], nx_w)[0], derivates)
            gradients[i] = [scalar_vector_product(d/len(examples), net[i].inputs) for d in delta[i]]

        SGD(weights, gradients, net)


    print(l)

    return weights


def NeuaralNetLeaner(dataset, epoch=2000):
    # init network
    net = [InputLayer(4), DenseLayer(4, 4), DenseLayer(4, 3)]
    # init loss
    loss = mse_loss
    # init data

    for e in range(epoch):
        print("epoch:", e)

        weights = [[node.weights for node in layer.nodes] for layer in net]

        theta = BackPropagation(dataset, weights, net, loss)

        # update the weights of network
        for i in range(len(net)):
            if theta[i]:
                for j in range(len(theta[i])):
                    net[i].nodes[j].weights = theta[i][j]








if __name__ == '__main__':
    iris = DataSet(name="iris")
    classes = ["setosa", "versicolor", "virginica"]
    iris.classes_to_numbers(classes)
    network = [InputLayer(4), DenseLayer(4,4), DenseLayer(4,3)]
    # loss = mse_loss
    # for _ in range(1):
    #     network = BackPropagationLearner(iris, network, loss)
    NeuaralNetLeaner(iris)
