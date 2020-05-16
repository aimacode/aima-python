import pytest
from keras.datasets import imdb

from deep_learning4e import *
from learning4e import DataSet, grade_learner, err_ratio

random.seed("aima-python")

iris_tests = [([5.0, 3.1, 0.9, 0.1], 0),
              ([5.1, 3.5, 1.0, 0.0], 0),
              ([4.9, 3.3, 1.1, 0.1], 0),
              ([6.0, 3.0, 4.0, 1.1], 1),
              ([6.1, 2.2, 3.5, 1.0], 1),
              ([5.9, 2.5, 3.3, 1.1], 1),
              ([7.5, 4.1, 6.2, 2.3], 2),
              ([7.3, 4.0, 6.1, 2.4], 2),
              ([7.0, 3.3, 6.1, 2.5], 2)]


def test_neural_net():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target

    X, y = (np.array([x[:n_features] for x in iris.examples]),
            np.array([x[n_features] for x in iris.examples]))

    nnl_gd = NeuralNetworkLearner(iris, [4], l_rate=0.15, epochs=100, optimizer=stochastic_gradient_descent).fit(X, y)
    assert grade_learner(nnl_gd, iris_tests) > 0.7
    assert err_ratio(nnl_gd, iris) < 0.15

    nnl_adam = NeuralNetworkLearner(iris, [4], l_rate=0.001, epochs=200, optimizer=adam).fit(X, y)
    assert grade_learner(nnl_adam, iris_tests) > 0.7
    assert err_ratio(nnl_adam, iris) < 0.15


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target

    X, y = (np.array([x[:n_features] for x in iris.examples]),
            np.array([x[n_features] for x in iris.examples]))

    pl_gd = PerceptronLearner(iris, l_rate=0.01, epochs=100, optimizer=stochastic_gradient_descent).fit(X, y)
    assert grade_learner(pl_gd, iris_tests) == 1
    assert err_ratio(pl_gd, iris) < 0.2

    pl_adam = PerceptronLearner(iris, l_rate=0.01, epochs=100, optimizer=adam).fit(X, y)
    assert grade_learner(pl_adam, iris_tests) == 1
    assert err_ratio(pl_adam, iris) < 0.2


def test_rnn():
    data = imdb.load_data(num_words=5000)

    train, val, test = keras_dataset_loader(data)
    train = (train[0][:1000], train[1][:1000])
    val = (val[0][:200], val[1][:200])

    rnn = SimpleRNNLearner(train, val)
    score = rnn.evaluate(test[0][:200], test[1][:200], verbose=False)
    assert score[1] >= 0.2


def test_autoencoder():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    inputs = np.asarray(iris.examples)

    al = AutoencoderLearner(inputs, 100)
    print(inputs[0])
    print(al.predict(inputs[:1]))


if __name__ == "__main__":
    pytest.main()
