"""Perception (Chapter 24)"""

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, InputLayer, Conv2D, MaxPooling2D
from keras.models import Sequential

from utils4e import gaussian_kernel_2D


# ____________________________________________________
# 24.3 Early Image Processing Operators
# 24.3.1 Edge Detection


def array_normalization(array, range_min, range_max):
    """Normalize an array in the range of (range_min, range_max)"""
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    array = array - np.min(array)
    array = array * (range_max - range_min) / np.max(array) + range_min
    return array


def gradient_edge_detector(image):
    """
    Image edge detection by calculating gradients in the image
    :param image: numpy ndarray or an iterable object
    :return: numpy ndarray, representing a gray scale image
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # gradient filters of x and y direction edges
    x_filter, y_filter = np.array([[1, -1]]), np.array([[1], [-1]])
    # convolution between filter and image to get edges
    y_edges = scipy.signal.convolve2d(image, x_filter, 'same')
    x_edges = scipy.signal.convolve2d(image, y_filter, 'same')
    edges = array_normalization(x_edges + y_edges, 0, 255)
    return edges


def gaussian_derivative_edge_detector(image):
    """Image edge detector using derivative of gaussian kernels"""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    gaussian_filter = gaussian_kernel_2D()
    # init derivative of gaussian filters
    x_filter = scipy.signal.convolve2d(gaussian_filter, np.asarray([[1, -1]]), 'same')
    y_filter = scipy.signal.convolve2d(gaussian_filter, np.asarray([[1], [-1]]), 'same')
    # extract edges using convolution
    y_edges = scipy.signal.convolve2d(image, x_filter, 'same')
    x_edges = scipy.signal.convolve2d(image, y_filter, 'same')
    edges = array_normalization(x_edges + y_edges, 0, 255)
    return edges


def laplacian_edge_detector(image):
    """Extract image edge with laplacian filter"""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    # init laplacian filter
    laplacian_kernel = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # extract edges with convolution
    edges = scipy.signal.convolve2d(image, laplacian_kernel, 'same')
    edges = array_normalization(edges, 0, 255)
    return edges


def show_edges(edges):
    """ helper function to show edges picture"""
    plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


# __________________________________________________
# 24.3.3 Optical flow


def sum_squared_difference(pic1, pic2):
    """SSD of two frames"""
    pic1 = np.asarray(pic1)
    pic2 = np.asarray(pic2)
    assert pic1.shape == pic2.shape
    min_ssd = np.inf
    min_dxy = (np.inf, np.inf)

    # consider picture shift from -30 to 30
    for Dx in range(-30, 31):
        for Dy in range(-30, 31):
            # shift the image
            shifted_pic = np.roll(pic2, Dx, axis=0)
            shifted_pic = np.roll(shifted_pic, Dy, axis=1)
            # calculate the difference
            diff = np.sum((pic1 - shifted_pic) ** 2)
            if diff < min_ssd:
                min_dxy = (Dx, Dy)
                min_ssd = diff
    return min_dxy, min_ssd


# ____________________________________________________
# segmentation

def gen_gray_scale_picture(size, level=3):
    """
    Generate a picture with different gray scale levels
    :param size: size of generated picture
    :param level: the number of level of gray scales in the picture,
                  range (0, 255) are equally divided by number of levels
    :return image in numpy ndarray type
    """
    assert level > 0
    # init an empty image
    image = np.zeros((size, size))
    if level == 1:
        return image
    # draw a square on the left upper corner of the image
    for x in range(size):
        for y in range(size):
            image[x, y] += (250 // (level - 1)) * (max(x, y) * level // size)
    return image


gray_scale_image = gen_gray_scale_picture(3)


def probability_contour_detection(image, discs, threshold=0):
    """
    Detect edges/contours by applying a set of discs to an image
    :param image: an image in type of numpy ndarray
    :param discs: a set of discs/filters to apply to pixels of image
    :param threshold: threshold to tell whether the pixel at (x, y) is on an edge
    :return image showing edges in numpy ndarray type
    """
    # init an empty output image
    res = np.zeros(image.shape)
    step = discs[0].shape[0]
    for x_i in range(0, image.shape[0] - step + 1, 1):
        for y_i in range(0, image.shape[1] - step + 1, 1):
            diff = []
            # apply each pair of discs and calculate the difference
            for d in range(0, len(discs), 2):
                disc1, disc2 = discs[d], discs[d + 1]
                # crop the region of interest
                region = image[x_i: x_i + step, y_i: y_i + step]
                diff.append(np.sum(np.multiply(region, disc1)) - np.sum(np.multiply(region, disc2)))
            if max(diff) > threshold:
                # change color of the center of region
                res[x_i + step // 2, y_i + step // 2] = 255
    return res


def group_contour_detection(image, cluster_num=2):
    """
    Detecting contours in an image with k-means clustering
    :param image: an image in numpy ndarray type
    :param cluster_num: number of clusters in k-means
    """
    img = image
    Z = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = cluster_num
    # use kmeans in opencv-python
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    # show the image
    # cv2.imshow('res2', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res2


def image_to_graph(image):
    """
    Convert an image to an graph in adjacent matrix form
    """
    graph_dict = {}
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            graph_dict[(x, y)] = [(x + 1, y) if x + 1 < image.shape[0] else None,
                                  (x, y + 1) if y + 1 < image.shape[1] else None]
    return graph_dict


def generate_edge_weight(image, v1, v2):
    """
    Find edge weight between two vertices in an image
    :param image: image in numpy ndarray type
    :param v1, v2: verticles in the image in form of (x index, y index)
    """
    diff = abs(image[v1[0], v1[1]] - image[v2[0], v2[1]])
    return 255 - diff


class Graph:
    """Graph in adjacent matrix to represent an image"""

    def __init__(self, image):
        """image: ndarray"""
        self.graph = image_to_graph(image)
        # number of columns and rows
        self.ROW = len(self.graph)
        self.COL = 2
        self.image = image
        # dictionary to save the maximum flow of each edge
        self.flow = {}
        # initialize the flow
        for s in self.graph:
            self.flow[s] = {}
            for t in self.graph[s]:
                if t:
                    self.flow[s][t] = generate_edge_weight(image, s, t)

    def bfs(self, s, t, parent):
        """Breadth first search to tell whether there is an edge between source and sink
        parent: a list to save the path between s and t"""
        # queue to save the current searching frontier
        queue = [s]
        visited = []

        while queue:
            u = queue.pop(0)
            for node in self.graph[u]:
                # only select edge with positive flow
                if node not in visited and node and self.flow[u][node] > 0:
                    queue.append(node)
                    visited.append(node)
                    parent.append((u, node))
        return True if t in visited else False

    def min_cut(self, source, sink):
        """Find the minimum cut of the graph between source and sink"""
        parent = []
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = np.inf
            # find the minimum flow of s-t path
            for s, t in parent:
                path_flow = min(path_flow, self.flow[s][t])

            max_flow += path_flow

            # update all edges between source and sink
            for s in self.flow:
                for t in self.flow[s]:
                    if t[0] <= sink[0] and t[1] <= sink[1]:
                        self.flow[s][t] -= path_flow
            parent = []
        res = []
        for i in self.flow:
            for j in self.flow[i]:
                if self.flow[i][j] == 0 and generate_edge_weight(self.image, i, j) > 0:
                    res.append((i, j))
        return res


def gen_discs(init_scale, scales=1):
    """
    Generate a collection of disc pairs by splitting an round discs with different angles
    :param init_scale: the initial size of each half discs
    :param scales: scale number of each type of half discs, the scale size will be doubled each time
    :return: the collection of generated discs: [discs of scale1, discs of scale2...]
    """
    discs = []
    for m in range(scales):
        scale = init_scale * (m + 1)
        disc = []
        # make the full empty dist
        white = np.zeros((scale, scale))
        center = (scale - 1) / 2
        for i in range(scale):
            for j in range(scale):
                if (i - center) ** 2 + (j - center) ** 2 <= (center ** 2):
                    white[i, j] = 255
        # generate lower half and upper half
        lower_half = np.copy(white)
        lower_half[:(scale - 1) // 2, :] = 0
        upper_half = lower_half[::-1, ::-1]
        # generate left half and right half
        disc += [lower_half, upper_half, np.transpose(lower_half), np.transpose(upper_half)]
        # generate upper-left, lower-right, upper-right, lower-left half discs
        disc += [np.tril(white, 0), np.triu(white, 0), np.flip(np.tril(white, 0), axis=0),
                 np.flip(np.triu(white, 0), axis=0)]
        discs.append(disc)
    return discs


# __________________________________________________
# 24.4 Classifying Images


def load_MINST(train_size, val_size, test_size):
    """Load MINST dataset from keras"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    total_size = len(x_train)
    if train_size + val_size > total_size:
        train_size = total_size - val_size
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    test_x = x_test.astype('float32')
    test_x /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return ((x_train[:train_size], y_train[:train_size]),
            (x_train[train_size:train_size + val_size], y_train[train_size:train_size + val_size]),
            (x_test[:test_size], y_test[:test_size]))


def simple_convnet(size=3, num_classes=10):
    """
    Simple convolutional network for digit recognition
    :param size: number of convolution layers
    :param num_classes: number of output classes
    :return a convolution network in keras model type
    """
    model = Sequential()
    # add input layer for images of size (28, 28)
    model.add(InputLayer(input_shape=(1, 28, 28)))
    # add convolution layers and max pooling layers
    for _ in range(size):
        model.add(Conv2D(32, (2, 2), padding='same', kernel_initializer='random_uniform'))
        model.add(MaxPooling2D(padding='same'))

    # add flatten layer and output layers
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model):
    """Train the simple convolution network"""
    # load dataset
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_MINST(1000, 100, 100)
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=5, verbose=2, batch_size=32)
    scores = model.evaluate(test_x, test_y, verbose=1)
    print(scores)
    return model


# _____________________________________________________
# 24.5 DETECTING OBJECTS


def selective_search(image):
    """
    Selective search for object detection
    :param image: str, the path of image or image in ndarray type with 3 channels
    :return list of bounding boxes, each element is in form of [x_min, y_min, x_max, y_max]
    """
    if not image:
        im = cv2.imread("./images/stapler1-test.png")
    elif isinstance(image, str):
        im = cv2.imread(image)
    else:
        im = np.stack(image * 3, axis=-1)

    # use opencv python to extract bounding box with selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

    # show bounding boxes with the input image
    image_out = im.copy()
    for rect in rects[:100]:
        print(rect)
        x, y, w, h = rect
        cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Output", image_out)
    cv2.waitKey(0)
    return rects


# faster RCNN
def pool_rois(feature_map, rois, pooled_height, pooled_width):
    """
    Applies ROI pooling for a single image and various ROIs
    :param feature_map: ndarray, in shape of (width, height, channel)
    :param rois: list of roi
    :param pooled_height: height of pooled area
    :param pooled_width: width of pooled area
    :return list of pooled features
    """

    def curried_pool_roi(roi):
        return pool_roi(feature_map, roi, pooled_height, pooled_width)

    pooled_areas = list(map(curried_pool_roi, rois))
    return pooled_areas


def pool_roi(feature_map, roi, pooled_height, pooled_width):
    """
    Applies a single ROI pooling to a single image
    :param feature_map: ndarray, in shape of (width, height, channel)
    :param roi: region of interest, in form of [x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio]
    :return feature of pooling output, in shape of (pooled_width, pooled_height)
    """

    # Compute the region of interest
    feature_map_height = int(feature_map.shape[0])
    feature_map_width = int(feature_map.shape[1])

    h_start = int(feature_map_height * roi[0])
    w_start = int(feature_map_width * roi[1])
    h_end = int(feature_map_height * roi[2])
    w_end = int(feature_map_width * roi[3])

    region = feature_map[h_start:h_end, w_start:w_end, :]

    # Divide the region into non overlapping areas
    region_height = h_end - h_start
    region_width = w_end - w_start
    h_step = region_height // pooled_height
    w_step = region_width // pooled_width

    areas = [[(
        i * h_step,
        j * w_step,
        (i + 1) * h_step if i + 1 < pooled_height else region_height,
        (j + 1) * w_step if j + 1 < pooled_width else region_width)
        for j in range(pooled_width)]
        for i in range(pooled_height)]

    # take the maximum of each area and stack the result
    def pool_area(x):
        return np.max(region[x[0]:x[2], x[1]:x[3], :])

    pooled_features = np.stack([[pool_area(x) for x in row] for row in areas])
    return pooled_features

# faster rcnn demo can be installed and shown in jupyter notebook
# def faster_rcnn_demo(directory):
#     """
#     show the demo of rcnn, the model is from
#     @inproceedings{renNIPS15fasterrcnn,
#     Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
#     Title = {Faster {R-CNN}: Towards Real-Time Object Detection
#              with Region Proposal Networks},
#     Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
#     Year = {2015}}
#     :param directory: the directory where the faster rcnn model is installed
#     """
# os.chdir(directory + '/lib')
# # make file
# os.system("make clean")
# os.system("make")
# # run demo
# os.chdir(directory)
# os.system("./tools/demo.py")
# return 0
