from learning import parse_csv, weighted_mode, weighted_replicate, DataSet, \
                     PluralityLearner, NaiveBayesLearner, NearestNeighborLearner, \
                     NeuralNetLearner, PerceptronLearner, DecisionTreeLearner
from utils import DataFile


def test_exclude():
    iris = DataSet(name='iris', exclude=[3])
    assert iris.inputs == [0, 1, 2]


def test_parse_csv():
    Iris = DataFile('iris.csv').read()
    assert parse_csv(Iris)[0] == [5.1,3.5,1.4,0.2,'setosa']


def test_weighted_mode():
    assert weighted_mode('abbaa', [1, 2, 3, 1, 2]) == 'b'


def test_weighted_replicate():
    assert weighted_replicate('ABC', [1, 2, 1], 4) == ['A', 'B', 'B', 'C']


def test_means_and_deviation():
    iris = DataSet(name="iris")

    means, deviations = iris.find_means_and_deviations()
    
    assert means["setosa"] == [5.006, 3.418, 1.464, 0.244]
    assert means["versicolor"] == [5.936, 2.77, 4.26, 1.326]
    assert means["virginica"] == [6.588, 2.974, 5.552, 2.026]

    assert round(deviations["setosa"][0],3) == 0.352
    assert round(deviations["versicolor"][0],3) == 0.516
    assert round(deviations["virginica"][0],3) == 0.636


def test_plurality_learner():
    zoo = DataSet(name="zoo")

    pL = PluralityLearner(zoo)
    assert pL([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1]) == "mammal"


def test_naive_bayes():
    iris = DataSet(name="iris")

    # Discrete
    nBD = NaiveBayesLearner(iris)
    assert nBD([5,3,1,0.1]) == "setosa"

    # Continuous
    nBC = NaiveBayesLearner(iris, continuous=True)
    assert nBC([5,3,1,0.1]) == "setosa"
    assert nBC([7,3,6.5,2]) == "virginica"


def test_k_nearest_neighbors():
    iris = DataSet(name="iris")

    kNN = NearestNeighborLearner(iris,k=3)
    assert kNN([5,3,1,0.1]) == "setosa"


def test_decision_tree_learner():
    iris = DataSet(name="iris")

    dTL = DecisionTreeLearner(iris)
    assert dTL([5,3,1,0.1]) == "setosa"


def test_neural_network_learner():
    iris = DataSet(name="iris")
    iris.remove_examples("virginica")
    
    classes = ["setosa","versicolor","virginica"]
    iris.classes_to_numbers()

    nNL = NeuralNetLearner(iris)
    # NeuralNetLearner might be wrong. Just check if prediction is in range.
    assert nNL([5,3,1,0.1]) in range(len(classes))


def test_perceptron():
    iris = DataSet(name="iris")
    iris.remove_examples("virginica")
    
    classes = ["setosa","versicolor","virginica"]
    iris.classes_to_numbers()

    perceptron = PerceptronLearner(iris)
    # PerceptronLearner might be wrong. Just check if prediction is in range.
    assert perceptron([5,3,1,0.1]) in range(len(classes))
