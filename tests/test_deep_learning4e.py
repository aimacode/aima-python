import pytest

from deep_learning4e import *
from learning4e import DataSet, grade_learner, err_ratio
from keras.datasets import imdb
import numpy as np

random.seed("aima-python")


def test_neural_net():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    nnl_adam = NeuralNetLearner(iris, [4], learning_rate=0.001, epochs=200, optimizer=adam_optimizer)
    nnl_gd = NeuralNetLearner(iris, [4], learning_rate=0.15, epochs=100, optimizer=gradient_descent)
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    assert grade_learner(nnl_adam, tests) >= 1 / 3
    assert grade_learner(nnl_gd, tests) >= 1 / 3
    assert err_ratio(nnl_adam, iris) < 0.21
    assert err_ratio(nnl_gd, iris) < 0.21


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    pl = PerceptronLearner(iris, learning_rate=0.01, epochs=100)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(pl, tests) > 1 / 2
    assert err_ratio(pl, iris) < 0.4


def test_rnn():
    data = imdb.load_data(num_words=5000)
    train, val, test = keras_dataset_loader(data)
    train = (train[0][:1000], train[1][:1000])
    val = (val[0][:200], val[1][:200])
    rnn = SimpleRNNLearner(train, val)
    score = rnn.evaluate(test[0][:200], test[1][:200], verbose=0)
    assert score[1] >= 0.3


def test_auto_encoder():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    inputs = np.asarray(iris.examples)
    al = AutoencoderLearner(inputs, 100)
    print(inputs[0])
    print(al.predict(inputs[:1]))


if __name__ == "__main__":
    pytest.main()
