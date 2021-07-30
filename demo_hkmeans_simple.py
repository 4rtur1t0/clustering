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

    # X, y = make_moons(n_samples=n_samples, noise=0.1)

    if data_type == np.uint8:
        X = np.round(X) + np.array([10, 10])
        X = X.astype(np.uint8)
    # Plot init seeds along side sample data
    plt.figure()
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
    plt.title("RESULT! Data clustered")
    plt.show()
    return X, y


def plot_mismatched(X, labels1, labels2, centroids):
    plt.rcParams['font.size'] = '12'

    plt.figure()
    colors = cycle(['r', 'g', 'b', 'c', 'm', 'y'])
    for k in range(len(set(labels2))):
        cluster_data = (labels2 == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=20)
        plt.scatter(centroids[k][0], centroids[k][1], c='k', marker='*', s=200)
        plt.annotate(str(k), (centroids[k][0], centroids[k][1]))
    # plt.show()

    mismatched_idx = np.where(np.abs(labels2 - labels1) > 0)
    lab1 = labels1[mismatched_idx]
    lab2 = labels2[mismatched_idx]
    # for all the mismatchs, annotate both clusters
    X = X[mismatched_idx]
    for i in range(len(lab1)):
        k1 = lab1[i]
        k2 = lab2[i]
        plt.scatter(X[i, 0], X[i, 1], c='k', marker='o', s=50, facecolors='none')
        # annotate both labels for that datapoint
        plt.annotate('bf' + str(k1) + '-- td' + str(k2) , (X[i, 0], X[i, 1]))
        # plt.annotate(str(k2), (X[i, 0], X[i, 1]))
    plt.show()
    print('debug')


if __name__ == '__main__':
    n_samples = 70
    n_samples_silhouette = int(np.round(0.2*n_samples))
    centers = 3

    kw = 2
    lw = 3
    print('Generating data: ')
    X, y = generate_data(n_samples=n_samples, centers=centers)

    kmeans_params = {'tol': 0.001,
                     'max_iter': 500,
                     'distance_function': 'euclidean',
                     'centroid_replacement': False,
                     'averaging_function': 'mean', #
                     'init_method': 'kmeans++',
                     'plot_progress': True}
    start_time = time.time()
    hk = HKMeans(kw=kw, lw=lw, kmeans_params=kmeans_params)
    # hk.print_tree()
    hk.fit(X)
    elapsed_time = time.time() - start_time
    hk.print_tree()
    hk.print_words()
    hk.plot_tree_data(X)

    print("Clustering data with size: ", len(X))
    print('Trained a tree with (kw, lw): ', kw, ', ', lw)
    print('Fit time: ', elapsed_time, '(s)')
    print('Total number of nodes: ', hk.get_number_of_nodes())
    print('Total number of leaves: ', hk.get_number_of_leaves())
    print('Total number of words expected: ', hk.get_expected_number_of_words())
    print('Total number of words found: ', hk.count_number_of_words())
    print('Total cost of fit (at all hierarchy levels): ', hk.get_total_cost())
    print('Number of datapoints per leaf (WORD): ', hk.get_n_datapoints_per_word())

    # use a brute force prediction: use the clos
    # est istance to each datapoint
    print("Brute force prediction: ")
    start_time = time.time()
    new_labels1, _ = hk.predict_brute_force(X)
    elapsed_time = time.time() - start_time
    print("Brute force took: ", elapsed_time)

    print("Top down prediction: ")
    start_time = time.time()
    new_labels2 = hk.predict_top_down(X)
    elapsed_time = time.time() - start_time
    print("top down took: ", elapsed_time)
    print("Num of errors in classification (a small number of mismatches is due to the approximation of the top-down classification): ", np.sum(np.abs(new_labels2 - new_labels1) > 0))
    sil_score = silhouette_score(X, new_labels1, sample_size=n_samples_silhouette)
    print("Silhouette Coefficient of tree classification: ", sil_score)

    plot_mismatched(X, new_labels1, new_labels2, hk.leaf_centroids)



