"""

A demo of a Vanilla k-means with 2-dimensional data.

Using:
    - Real numbers.
    - An Euclidean distance function.
    - A standard averaging function.

    Best results are obtained when using the k-means++ initialization method.
    Use the option plot_progress=True to view the movement and assignation of the centroids at each algorithm step.

Some inspiration:
    https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/
    https://gist.github.com/wildonion/245ca496293ee537214b3dbbb3c558c9

"""
import json
import numpy as np
import time

from cluster.kmeans import KMeans


def generate_data(n_samples=250, centers=2, data_type=np.float64):
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs, make_moons, make_hastie_10_2
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      cluster_std=0.50,
                      random_state=1)

    if data_type == np.uint8:
        X = np.round(X) + np.array([10, 10])
        X = X.astype(np.uint8)
    # Plot init seeds along side sample data
    plt.figure(1)
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    for k, col in enumerate(colors):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1],
                    c=col, marker='.', s=10)
    # plt.scatter(centers_init[:, 0], centers_init[:, 1], c='b', s=50)
    plt.title("Data generated")
    plt.show()
    return X, y


if __name__ == '__main__':
    n_samples = 1500
    centers = 5
    X, y = generate_data(n_samples=n_samples, centers=centers)

    clf = KMeans(k=centers, tol=0.001, max_iter=500,
                 distance_function='euclidean',
                 centroid_replacement=False,
                 init_method='kmeans++',
                 plot_progress=True)
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


