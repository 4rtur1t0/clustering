"""
A vanilla k-medoids algorithm.


1-->  entender ejemplo
https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05

2-- comprobar
https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/partitioning_around_medoids.py


3 --> Revisar estos papers
https://link.springer.com/content/pdf/10.1007/s10846-020-01230-z.pdf

https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417400X00905/1-s2.0-S0957417402001859/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIGp5Oz9Ym6NfDed997ybS%2BON2c4rnWWdM1Nqa0dui4sHAiBLUFggj0ibDrDDnh0eB%2FT%2B7pfIZ9A%2FyHA7PH4M3L3%2B8CqDBAjV%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAQaDDA1OTAwMzU0Njg2NSIMbR0xectNehv%2Be%2F6MKtcDYI%2B2k5LcuPKOo%2B4NAX8n3%2BwBDb%2BppBB1%2F6OG465gL9mAIWbubdRBhIj7PIqT5itDLtloJ2ueI%2BYZ7bb2RGUBy6VBmwuGeV%2BcV5T9VhANHeLhAVJy3K2dWVUkIloEr5oIMHYT61S4j86WrYQQVtbTa1rtfuKBG%2F9renlH99oHpi7tjHbbetTnBT%2Bmbg13llyD18ncC6IiYNNojiK2wXG3Rt%2FGFua9JDPyGvW6ny1r8hiYfF6S254EKf3nJgd3KNhBqN3hlggXSg0jeXsLKUWeUCTgPy1GoZ6Pf4o5dOx0WtsCZ%2F2tnm%2FD2MPJnxqeclO%2F2mD0qMg8bHWo%2Fi0KfB5%2FbbYKAgv0iM6QtyUKrSO5w9Vn0JiVWDQVcTva3sZVD0Jw1lPBC6a6lwVzU%2FXJDZOT%2B%2FuKD%2Bhoq6wEU7IrL7wK4JuBFVGkAsOh6u1VOKi%2F4H6T4BI1bE9hPo7%2FSQ7HuA8vYJIsXY7MvjOiQJ272VR9493NtCjzsRrNlvW0QuSAOL4fw76SLZvV0TANpBPrR322aU6awNQ0xLyEV8hlAhyCcYWzL2kFjHKR9yfebIy7okLxUG%2FbecI9hKUmunnrVs4eHPJqREdh3gQdbFyLBNiVRZKT4XmIOOZNMIf1%2B4YGOqYBILBhqXoZz5awS1gcRjrlvlso2neTCHLQqMJs%2BS7qYMVXoPy03QCfZribfpl5sUNBTVq2J03bS1aROIoJVI7GdjxPyWnV1zGkIC6%2FSQB2yXqdsbJX3NT61O9lDdnWCLLOUo1DnZ4Mx2KJsUincsjqeEv6uoh6cPKs7VGKP1WPK7YiDUW87H1JO3P%2BmZctavBL12X0pdgFyIrhBgu6NSW5hUTUrU%2FNGQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210702T120513Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYN7MTX6X%2F20210702%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e0cefac661fec2be15a90a569ba2b364fa810dd7d68bee103ec46bf4e413e7c3&hash=513c045afb1fd325a2e26fadfefb0b452e7520051eafb038e55b84068fe06823&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417402001859&tid=spdf-b466b110-99f1-4f21-9b7b-2572228dc6ac&sid=c8017ae2588a164a2b2978d74f6e637dac49gxrqb&type=client


https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/?completed=/mean-shift-titanic-dataset-machine-learning-tutorial/

https://pythonprogramming.net/hierarchical-clustering-mean-shift-machine-learning-tutorial/?completed=/k-means-from-scratch-2-machine-learning-tutorial/
https://medium.com/analytics-vidhya/supernaive-k-medoids-clustering-31db7bfc5075

https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/dbscan.py

https://www.sciencedirect.com/science/article/pii/S095741740800081X

https://www.geeksforgeeks.org/ml-k-medoids-clustering-with-example/


Aggregating Binary Local Descriptors
for Image Retrieval
https://arxiv.org/pdf/1608.00813.pdf

http://www.nmis.isti.cnr.it/falchi/Draft/2016-VISAPP-BinaryFeatures-DRAFT.pdf
"""
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
# import json
style.use('ggplot')
import numpy as np

# _nbits[k] is the number of 1s in the binary representation of k for 0 <= k < 256.
_nbits = np.array(
      [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
       4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
       4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
       3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
       4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
       5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
       3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
       3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
       4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
       6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
       5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
       7, 7, 8], dtype=np.uint8)


def hamming(a, b):
    """
    Compute a hamming distance between descriptors a and b
    Use a bitwise or on the whole array of bytes.
    Next, _nbits is a precomputed array that stores the number of bits on each 1-byte result.
    :param a:
    :param b:
    :return:
    """
    a = np.uint8(a)
    b = np.uint8(b)
    c = np.bitwise_xor(a, b)
    n = _nbits[c].sum()
    return n


class KMedoids():
    def __init__(self, k=5, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.cost = 0
        self.labels = []
        self.centroids = {}
        self.distances = []
        # number of datapoints
        self.N = None

    def fit(self, data):
        """
        Main iteration loop of the algorithm.
        A set of k centroids is selected.
        Next, each datapoint is assigned to the centroid that minimizes its distance.
        Each centroid is then updated with the mean value of all its datapoints.

        The process is repeated until convergence is found in the average or sum of distances to all clusters.
        :param data:
        :return:
        """
        self.N = len(data)
        # Init k centroids (medoids) randomly
        indexes = np.random.randint(0, len(data)-1, self.k) # select k indices from all data in dataset
        self.centroids = data[indexes]

        # for i in range(self.k):
        #     self.centroids[i] = data[i]
        # perform max iterations of the algorithm
        for i in range(self.max_iter):
            print("Iteration number: ", i)
            cost = self.fit_iteration(data)
            # finally, this checks whether we have converged.
            # convergence is found if the improvement in the total cost (inertia) is low
            converged = self.has_converged(cost)
            if converged:
                print("Converged!")
                break
        if i == (self.max_iter-1):
            print("Caution: max number of iterations reached without convergence")

    def fit_iteration(self, data):
        """
        A iteration of the kmeans algorithm
         maintain a list where each datapoint points to one of the clusters.

        assign each datapoint to the closest centroid (assign each datapoint to a cluster)
        each label[i] stores the cluster associated to the datapoint data[i]
        No copying of datapoints to other variables is done, thus reducing the amount of memory required.
        :param data:
        :return:
        """
        #
        [self.labels, self.distances] = self.predict(data)
        # [self.labels, self.distances] = self.predict2(data)
        self.update_centroids(data)
        # compute the total cost of the current assigment based on the previous distances
        cost = np.sum(self.distances)
        # cost = np.average(self.distances)
        return cost

    def predict(self, data):
        """
        For each datapoint, we assign it to a cluster.
        Being interpreted, this forces python to iterate over all datapoints and the method is slow.
        :param data:
        :return:
        """
        labels = -1 * np.ones(len(data), dtype=np.int8)
        distances = np.zeros(len(data))
        for i in range(0, len(data)):
            # get datapoint i
            datapoint = data[i]
            # compute the distance of each datapoint to all clusters,find closest one and assign to it
            distances_to_centroids = []
            for k in range(0, self.k):
                # compute the hamming distance between datapoint i and cluster k
                distances_to_centroids.append(hamming(datapoint, self.centroids[k]))
                # distances_to_centroids.append(np.linalg.norm(datapoint - self.centroids[centroid]))
            # find min distance and its index (cluster index)
            min_dist = np.amin(distances_to_centroids)
            k_i = np.where(distances_to_centroids == min_dist)
            # a list is maintained with the clusters and distances that have been assigned to each datapoint
            labels[i] = int(k_i[0][0])
            distances[i] = min_dist
        return labels, distances

    def predict2(self, data):
        """
        Try to speed the assignation of datapoints to clusters.
        In order to do this we compute the distance of each cluster to all the datapoints.
        Next, the min distance datapoint-cluster is used for the assignation.
        The assignation is based
        :param data:
        :return:
        """
        distance_mat = []
        labels = -1*np.ones(len(data), dtype=np.int8)
        # find the distances of all datapoints to each cluster k
        # for each cluster, the operation computes the distance to all datapoints
        for k in range(0, self.k):
            centroid = self.centroids[k]
            # distances of all datapoints to centroid k
            d = np.linalg.norm(data - centroid, axis=1)
            distance_mat.append(d)
        distance_mat = np.array(distance_mat)
        datapoint_distances = np.zeros(len(data))
        # next, we select the
        for i in range(0, len(data)):
            col = distance_mat[:, i]
            min_dist = np.amin(col)
            k_i = np.where(col == min_dist)
            labels[i] = k_i[0][0]
            datapoint_distances[i] = min_dist
        return labels, datapoint_distances

    def update_centroids(self, data):
        """
        find the position of the new centroids.
        in the vanilla k-means this is done by averaging all the datapoints assigned to each cluster
        In the k-medoids algorithm, a voting scheme is deployed. This is generally called k-majority voting. As a result
        the new cluster

        Updating:
         1) iterate along along the datapoints of each cluster.
         2) consider that each datapoint may be a new medoid
         3) compute a cost for that new medoid.
         4) Select the minimum cost
         requires the computation of N distances per cluster
        :param data:
        :return:
        """
        for k in range(0, self.k):
            # find indexes that have been assigned to cluster i
            indexes = np.where(self.labels == k)
            # get data that corresponds to these indexes
            self.centroids[k] = np.average(data[indexes], axis=0)

    def has_converged(self, cost):
        """
        check whether inertia changes from last value
        cost is the current computed cost for the current cluster assignment.
        self.cost is the previous cost
        :return:
        """
        percent = 100.0*np.abs(cost-self.cost)/cost
        print("Percent in cost improved: ", percent)
        # update last cost
        self.cost = cost
        if percent < self.tol:
            return True
        return False

    def plot(self, X):
        """
        Caution, plotting only features 1 and 2 of the dataset.
        Data X is predicted and then plotted
        :param X:
        :return:
        """
        [labels, _] = self.predict(X)
        # plot current centroids
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
                        marker="o", color="k", s=150, linewidths=5)

        colors = 10 * ["g", "r", "c", "b", "k"]
        for label in set(labels.tolist()):
            color = colors[label]
            indexes = np.where(labels == label)
            plot_data = X[indexes]
            for featureset in plot_data:
                plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        plt.show()





