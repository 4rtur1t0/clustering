"""

"""
import json
import numpy as np
import time

from sklearn.cluster import AgglomerativeClustering

from cluster.kmeans import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == '__main__':
    n_clusters = 20

    # Loading 1000 ORB descriptors from file
    # X = load_data(1, 200)
    # in this case, the averaging makes sense if the bits are unpacked.
    # the average and discretization are a form of k-majority voting
    # print("Converting to long binary descriptors")
    # X = np.unpackbits(X, axis=1)

    X, _ = make_blobs(n_samples=10, centers=3, n_features=2, cluster_std=0.30, random_state=0)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=1.0,
                                         linkage='average')

    print("Using data size: ", len(X))
    start_time = time.time()
    clustering.fit(X)
    # clf.encode_centroids()
    elapsed_time = time.time() - start_time
    print('Fit time: ', elapsed_time, '(s)')
    # print("Current cost is: ", clustering.cost)
    print("Centroids are: ")
    # for c in clustering.centroids:
    #     print(clustering.centroids[c])


    # plot current centroids
    # for centroid in self.centroids:
    #     plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
    #                 marker="o", color="k", s=150, linewidths=5)



    plt.figure(2)
    plot_dendrogram(clustering, truncate_mode='level', p=3)


    plt.figure(1)

    colors = 10 * ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    for k in range(0, len(set(clustering.labels_))):
        cluster_data = (clustering.labels_ == k)
        col = colors[k]
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1],
                    c=col, marker='.', s=30)
    plt.show()


    # # pack data and also the resulting centroids
    # X = np.packbits(X, axis=1)
    # clustering.pack_centroids()
    #
    # # classify and plot
    # clustering.plot(X, 'Final result!')



    # clf.plot(X, 'Final result!')

    # distance_mat = np.zeros((20, 20))
    # for i in range(n_clusters):
    #     for j in range(n_clusters):
    #         centroid_i = clf.centroids[i]
    #         centroid_j = clf.centroids[j]
    #         d = np.linalg.norm(centroid_i-centroid_j)
    #         distance_mat[i, j] = d

    # print(distance_mat)
