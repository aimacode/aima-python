import random

import pytest

from perception4e import *
from PIL import Image
import numpy as np
import os

random.seed("aima-python")


def test_array_normalization():
    assert list(array_normalization([1, 2, 3, 4, 5], 0, 1)) == [0, 0.25, 0.5, 0.75, 1]
    assert list(array_normalization([1, 2, 3, 4, 5], 1, 2)) == [1, 1.25, 1.5, 1.75, 2]


def test_sum_squared_difference():
    image = Image.open(os.path.abspath("./images/broxrevised.png"))
    arr = np.asarray(image)
    arr1 = arr[10:500, :514]
    arr2 = arr[10:500, 514:1028]
    assert sum_squared_difference(arr1, arr1)[1] == 0
    assert sum_squared_difference(arr1, arr1)[0] == (0, 0)
    assert sum_squared_difference(arr1, arr2)[1] > 200000


def test_gen_gray_scale_picture():
    assert list(gen_gray_scale_picture(size=3, level=3)[0]) == [0, 125, 250]
    assert list(gen_gray_scale_picture(size=3, level=3)[1]) == [125, 125, 250]
    assert list(gen_gray_scale_picture(size=3, level=3)[2]) == [250, 250, 250]
    assert list(gen_gray_scale_picture(2, level=2)[0]) == [0, 250]
    assert list(gen_gray_scale_picture(2, level=2)[1]) == [250, 250]


def test_generate_edge_weight():
    assert generate_edge_weight(gray_scale_image, (0, 0), (2, 2)) == 5
    assert generate_edge_weight(gray_scale_image, (1, 0), (0, 1)) == 255


def test_graph_bfs():
    graph = Graph(gray_scale_image)
    assert not graph.bfs((1, 1), (0, 0), [])
    parents = []
    assert graph.bfs((0, 0), (2, 2), parents)
    assert len(parents) == 8


def test_graph_min_cut():
    image = gen_gray_scale_picture(size=3, level=2)
    graph = Graph(image)
    assert len(graph.min_cut((0, 0), (2, 2))) == 4
    image = gen_gray_scale_picture(size=10, level=2)
    graph = Graph(image)
    assert len(graph.min_cut((0, 0), (9, 9))) == 10


def test_gen_discs():
    discs = gen_discs(100, 2)
    assert len(discs) == 2
    assert len(discs[1]) == len(discs[0]) == 8


def test_simple_convnet():
    train, val, test = load_MINST(1000, 100, 10)
    model = simple_convnet()
    model.fit(train[0], train[1], validation_data=(val[0], val[1]), epochs=5, verbose=2, batch_size=32)
    scores = model.evaluate(test[0], test[1], verbose=1)
    assert scores[1] > 0.2


def test_ROIPoolingLayer():
    # Create feature map input
    feature_maps_shape = (200, 100, 1)
    feature_map = np.ones(feature_maps_shape, dtype='float32')
    feature_map[200 - 1, 100 - 3, 0] = 50
    roiss = np.asarray([[0.5, 0.2, 0.7, 0.4], [0.0, 0.0, 1.0, 1.0]])
    assert pool_rois(feature_map, roiss, 3, 7)[0].tolist() == [[1, 1, 1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 1, 1, 1]]
    assert pool_rois(feature_map, roiss, 3, 7)[1].tolist() == [[1, 1, 1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 1, 1, 50]]


if __name__ == '__main__':
    pytest.main()
