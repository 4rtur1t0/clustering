"""

A demo of the k-means with ORB descriptors computed on a set of images.

In this case, a hierarchical clustering is performed.

Using:
    - Binary ORB descriptors numbers.
    - A Hamming distance function to compute the distance between two descriptors.
    - An average mean function to find each cluster center.

    Best results are obtained when using the k-means++ initialization method.
    Use the option plot_progress=True to view the movement and assignation of the centroids at each algorithm step.

"""
import json
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
    all_descriptors = read_descriptors('data/descriptors.json')
    all_descriptors = flatten_list(all_descriptors)
    if max_index is None:
        max_index = len(all_descriptors)
    if sampling is None:
        sampling = 1
    all_descriptors = all_descriptors[0:max_index:sampling]
    X = np.array(all_descriptors, np.uint8)
    return X


if __name__ == '__main__':
    # Loading 1000 ORB descriptors from file
    X = load_data(1, 1000)

    clf = KMeans(k=100, tol=0.001, max_iter=500,
                 # distance_function='hamming',
                 averaging_function='mean-round',
                 centroid_replacement=False,
                 init_method='kmeans++',
                 plot_progress=False)
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
    clf.plot(X, 'Final result!')


