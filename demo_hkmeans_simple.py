"""

A demo of a Hierarchical K-means with:
- Kw branches and
- Lw levels.

Using the HKMeans class, which constructs constructs a Tree with a root node, each node has Kw children.
The total depth of the tree is Lw. Each node implements a kmeans algorithm on its own data.

"""
from itertools import cycle

import numpy as np
import time
from cluster.hkmeans import HKMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_hastie_10_2
from sklearn.metrics import silhouette_score


def generate_data(n_samples=250, centers=2, data_type=np.float64):
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      cluster_std=0.10,
                      random_state=0)
    if data_type == np.uint8:
        X = np.round(X) + np.array([10, 10])
        X = X.astype(np.uint8)
    # Plot init seeds along side sample data
    plt.figure(1)
    colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
    for k in range(centers):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=20)
    plt.title("Data generated")
    plt.show()
    return X, y


def plot_result(X, y, centroids):
    # Plot init seeds along side sample data
    plt.figure(1)
    colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
    for k in set(y):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=20)
        plt.scatter(centroids[k][0], centroids[k][1], c='k', marker='.', s=200)
    plt.title("Data clustered")
    plt.show()
    return X, y


if __name__ == '__main__':
    n_samples = 5000
    centers = 5
    kw = 2
    lw = 1
    X, y = generate_data(n_samples=n_samples, centers=centers)

    kmeans_params = {'tol': 0.001, 'max_iter': 500,
                   'distance_function': 'euclidean',
                   'centroid_replacement': False,
                   'init_method': 'kmeans++',
                   'plot_progress': True}

    hk = HKMeans(kw=kw, lw=lw, kmeans_params=kmeans_params)
    hk.print_tree()

    start_time = time.time()
    hk.fit(X)
    elapsed_time = time.time() - start_time
    hk.print_tree()
    hk.print_words()


    # hk.plot_tree_data(X)
    print("Clustering data with size: ", len(X))
    print('Trained a tree with (kw, lw): ', kw, ', ', lw)
    print('Total number of nodes: ', hk.get_number_of_nodes())
    print('Total number of leaves: ', hk.get_number_of_leaves())
    print('Total number of words expected: ', hk.get_expected_number_of_words())
    print('Total number of words found: ', hk.count_number_of_words())
    print('Total cost of fit (at last hierarchy level): ', hk.get_total_cost())
    print('Mean and variance in the number of datapoints per leaf: ', hk.get_n_datapoints_per_leaf_distribution())
    print('Mean and variance in the number of datapoints per Word: ', hk.get_n_datapoints_per_word_distribution())
    print('Fit time: ', elapsed_time, '(s)')
    print("Silhouette Coefficient: %0.3f" % silhouette_score(X, hk.labels, sample_size=1000))
    plot_result(X, hk.labels, hk.centroids)

    # predict the same data
    new_labels = hk.predict(X)




