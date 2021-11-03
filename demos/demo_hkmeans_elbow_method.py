"""
A demo of a Hierarchical K-means with the elbow method.

Different fits are performed by changing the depth level Lw. The parameters of the hierarchical K-means should be set by
looking at the resulting figure and observing the level Lw for which the cost is not reduced significantly. That is,
observe the ratio of change in the cost and select Lw where this ratio stabilizes.

"""
from itertools import cycle

import numpy as np
import time
from cluster.hkmeans import HKMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_hastie_10_2


def generate_data(n_samples=250, centers=2, data_type=np.float64):
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      cluster_std=0.10,
                      random_state=0)
    plt.figure(0)
    colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
    for k in range(centers):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=20)
    plt.title("Simulated data")
    plt.show(block=False)
    return X, y


if __name__ == '__main__':
    # simulated data
    n_samples = 10000
    centers = 256

    kw = 2
    X, y = generate_data(n_samples=n_samples, centers=centers)

    costs = []
    # construct trees with different depths
    depths = [2, 3, 4, 5, 6, 7, 8, 9]
    # repeat HKmeans with different lw (branching factor kw is assumed equal)
    for lw in depths:
        hk = HKMeans(kw=kw, lw=lw)
        # hk.create_tree()
        # hk.print_tree()
        print("Using data size: ", len(X))
        start_time = time.time()
        hk.fit(X)
        elapsed_time = time.time() - start_time
        hk.print_tree()
        print("Clustering data with size: ", len(X))
        print('Trained a tree with (kw, lw): ', kw, ', ', lw)
        print('Total number of nodes: ', hk.get_number_of_nodes())
        print('Total number of leaves: ', hk.get_number_of_leaves())
        print('Total number of words expected: ', hk.get_expected_number_of_words())
        print('Total number of words found: ', hk.count_number_of_words())
        print('Total cost of fit (at last hierarchy level): ', hk.get_total_cost())
        print('Mean and variance in the number of datapoints per leaf: ', hk.get_n_datapoints_distribution())
        # hk.check_special_case()
        print('Fit time: ', elapsed_time, '(s)')
        costs.append(hk.get_total_cost())

    hk.plot_tree_data(X)

        # print("Current silouhete is: ", clf.cost)
    plt.figure(1)
    plt.plot(depths, costs)
    plt.title('Cost of the tree for different levels lw')
    plt.show()




