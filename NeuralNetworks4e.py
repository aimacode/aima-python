import math
import statistics
from utils4e import sigmoid, dotproduct, softmax1D, conv1D, GaussianKernel, element_wise_product, \
    vector_add, random_weights, scalar_vector_product, matrix_multiplication, transpose2D, leaky_relu
from learning4e import DataSet
import random


def cross_entropy_loss(X, Y):
    n = len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def mse_loss(X, Y):
    n = len(X)
    return (1.0/2)*sum((x-y)**2 for x, y in zip(X, Y))


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


def SGD():
    """
    # use sgd to update learnable parameters
    """

    def update(theta, gradients, lrate=0.01):
        # theta = vector_add(theta, scalar_vector_product(-lrate, gradients))
        # return theta
        for i in range(len(gradients)):
            if gradients[i]:
                for j in range(len(gradients[i])):
                    theta[i][j] = list(vector_add(theta[i][j], list(
                        scalar_vector_product(-lrate, gradients[i][j]))))
    return update


def adam_optimizer(s, r, t, rho=(0.9, 0.999), delta=1/10**8):
    def update(theta, gradients, lrate=0.001):
        for i in range(1, len(gradients)):
            for j in range(len(gradients[i])):
                s[i][j] = vector_add(scalar_vector_product(rho[0],s[i][j]), scalar_vector_product((1-rho[0]),gradients[i][j]))
                r[i][j] = vector_add(scalar_vector_product(rho[1],r[i][j]),
                                     scalar_vector_product((1-rho[1]), element_wise_product(gradients[i][j], gradients[i][j])))
                s_hat = scalar_vector_product(1/(1-rho[0]**t), s[i][j])
                r_hat = scalar_vector_product(1/(1-rho[1]**t), r[i][j])
                delta_theta = [-lrate*s_x*1/(math.sqrt(r_x)+delta) for s_x, r_x in zip(s_hat, r_hat)]
                theta[i][j] = list(vector_add(theta[i][j], delta_theta))

    return update


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


def BackPropagation(inputs, targets, theta, net, loss, lrate=0.5, optimizor=SGD()):
    """The back-propagation algorithm for multilayer networks for only one epoch"""
    # Initialise weights outside of backprop

    assert len(inputs) == len(targets)
    o_units = len(net[-1].nodes)
    n_layers = len(net)
    batch_size = len(inputs)

    gradients = [[[0]*len(node.weights) for node in layer.nodes] for layer in net]

    batch_loss = 0
    for e in range(batch_size):
        i_val = inputs[e]
        t_val = targets[e]

        # Forward pass
        for i in range(1, n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        # Initialize delta
        delta = [[] for _ in range(n_layers)]
        # compute loss
        batch_loss += loss(t_val, layer_out)

        # Backward pass
        previous = [layer_out[i]-t_val[i] for i in range(o_units)]
        h_layers = n_layers - 1
        for i in range(h_layers, 0, -1):
            layer = net[i]
            derivative = [layer.activation.derivative(node.val) for node in layer.nodes]
            delta[i] = element_wise_product(previous, derivative)
            previous = matrix_multiplication([delta[i]], theta[i])[0]
            gradients[i] = [scalar_vector_product(d, net[i].inputs) for d in delta[i]]

        optimizor(theta, gradients)
        for i in range(len(net)):
            if theta[i]:
                for j in range(len(theta[i])):
                    net[i].nodes[j].weights = theta[i][j]
    print("loss:", batch_loss)
    return theta


def NeuaralNetLeaner(dataset, lrate=0.15, epoch=1000):
    # init network
    net = [InputLayer(4), DenseLayer(4, 4), DenseLayer(4, 3)]
    # init loss
    loss = mse_loss
    # init data
    examples =dataset.examples
    # inputs, targets = init_examples(examples, dataset.inputs, dataset.target, len(net[-1].nodes))

    s = [[[0]*len(node.weights) for node in layer.nodes] for layer in net]
    r = [[[0]*len(node.weights) for node in layer.nodes] for layer in net]

    for e in range(epoch):
        inputs, targets = init_examples(examples, dataset.inputs, dataset.target, len(net[-1].nodes))
        print("epoch:", e)
        opt = adam_optimizer(s, r, e+1)
        weights = [[node.weights for node in layer.nodes] for layer in net]
        theta = BackPropagation(inputs, targets, weights, net, loss, lrate=lrate)

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
    NeuaralNetLeaner(iris)
