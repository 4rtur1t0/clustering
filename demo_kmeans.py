"""

A Vanilla KMeans Algorithm.

Should be tested for speed.

Classifications considers an index vector and does not copy the data.

Further improvements could add a different distance function

https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/

https://gist.github.com/wildonion/245ca496293ee537214b3dbbb3c558c9
predict --> 10000 datos --> 5.94 s
"""
# import pandas as pd

# import matplotlib.pyplot as plt
# from matplotlib import style
import json
# style.use('ggplot')
import numpy as np
import time

from cluster.kmeans import KMeans


def read_descriptors(path):
    descriptors_input_path = path # + '/mav0/descriptors.json'
    with open(descriptors_input_path, 'r') as json_file:
        data = json.load(json_file)
        desc1 = data['descriptors_front']
        desc2 = data['descriptors_back']
        # extend list 1
        desc1.extend(desc2)
        return desc1


def flatten_list(all_descriptors):
    """
    Flattens list of descriptors: original list is of size
        n_landmarks list with [1500 x 32] in each
    """
    ret_descriptors = []
    for descriptors_image in all_descriptors:
        for desc in descriptors_image:
            ret_descriptors.append(desc)
    return ret_descriptors


def load_data(sampling=None, max_index=None):
    all_descriptors = []
    # load all descriptors and form a cluster
    # for path in paths:
    all_descriptors = read_descriptors('data/descriptors.json')
        # all_descriptors.extend(desc)
    all_descriptors = flatten_list(all_descriptors)
    if max_index is None:
        max_index = len(all_descriptors)
    if sampling is None:
        sampling = 1
    all_descriptors = all_descriptors[0:max_index:sampling]
    X = np.array(all_descriptors, np.uint8)
    return X



if __name__ == '__main__':

    # X = np.array([[1, 2],
    #               [1.5, 1.8],
    #               [5, 8],
    #               [8, 8],
    #               [1, 0.6],
    #               [9, 11],
    #               [1, 3],
    #               [8, 9],
    #               [0, 3],
    #               [5, 4],
    #               [6, 4],])

    X = np.array([[0, 0, 0],
                  [1, 1, 0],
                  [2, 0, 0],
                  [3, 0, 0],
                  [4, 0, 0],
                  [5, 0, 0],
                  [6, 0, 0],
                  [7, 0, 0],
                  [8, 0, 0],
                  [9, 0, 0],
                  [10, 0, 0],
                  [11, 0, 0],
                  [12, 0, 0],
                  [13, 0, 0],
                  [14, 0, 0], ])

    # X = load_data(1, 10000)

    clf = KMeans()
    print("Using data size: ", len(X))
    start_time = time.time()
    clf.fit(X)
    elapsed_time = time.time() - start_time
    print('Fit time: ', elapsed_time, '(s)')
    print("Current cost is: ", clf.cost)
    print("Centroids are: ")
    for c in clf.centroids:
        print(clf.centroids[c])
    # classify and plot
    clf.plot(X)


