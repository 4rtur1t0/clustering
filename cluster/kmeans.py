"""

A Vanilla KMeans Algorithm.

Parameters:
    k: number of clusters to form.
    tol: the tolerance to stop the algorithm whenever the improvement in cost is low.
    max_iter: the maximum number of iterations to perform. A warning is issued whenever this value is reached wihout
                convergence of the algorithm.

Options:
    Several options are available.
        distance_function: two distance functions are available Euclidean and Hamming distances


Typical usage:
    binary descriptors
    Hamming distance.
    Averaging with k-majority voting
    Use replacement of the centroid.

When using Hamming distance, mean-round must be used and c

clf = KMeans(k=3, tol=0.01, max_iter=500,
                 distance_function='hamming',
                 averaging_function='mean-round',
                 centroid_replacement=True)

Should be tested for speed.

Classifications considers an index vector and does not copy the data.

Further improvements could add a different distance function

https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/

https://gist.github.com/wildonion/245ca496293ee537214b3dbbb3c558c9
predict --> 10000 datos --> 5.94 s
"""
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import json

style.use('ggplot')
import numpy as np
import random

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

    Example:
        The number 13 is represented by 00001101.
        Likewise, 17 is represented by 00010001.
        The bit-wise XOR of 13 and 17 is therefore 00011100, or 28:

        >>> np.bitwise_xor(13, 17)
            28
        >>> np.binary_repr(28)
            '11100'

    Using bitwise_xor to compoute the operation. For example, given two arrays
    >>> np.bitwise_xor([31, 3, 67, 78], [5, 6, 90, 255])
    array([ 26,   5,  25, 177])

    :param a: a binary descriptor.
    :param b:
    :return:
    """
    # a = np.uint8(a)
    # b = np.uint8(b)
    c = np.bitwise_xor(a, b)
    n = _nbits[c].sum()
    return n


class KMeans():
    """
    Parameters:
        k: Number of clusters
        tol: The algorithm will stop iterating when the cost (in percentage) is improved below this value.
        max_iter: Max. iterations. The max number of iterations to perform.
    Options:
        - 'init': Initialization of clusters.
            - 'random': Select k cluster centers at random
            - 'kmeans++': kmeans++ initialization.
            - 'k-first': just use the first k samples in the dataset to initialize the algorithm.
        - 'distance': Distance function to compute distance between datapoints and cluster centers.
            - Euclidean L2-norm.
            - Hamming L1-norm.
        - 'averaging': Averaging of the clusters.
            - 'mean' (default): compute a standard averaging mean and leave as is. The new cluster centers are the mean
            of all datapoints assigned to that cluster.
            - 'mean-round':  compute a standard averaging mean and round to the nearest integer. The new cluster centers
             are the mean of all datapoints assigned to that cluster approximated to the nearest integer.
            - 'k-majority': This is an implementation of a vanilla k-majority voting. After averaging, form a discrete
            descriptor, each bit of each datapoint votes. The new cluster centers are computed with the 1 and 0 voted
            by the majority of their datapoints. This makes only sense for binary descriptors. This is equivalent to finding
            an average value over the 1 and 0 and, next, round the result to the closest 1 or 0.
        - mean_replacement: Replacing of the mean. After averaging each cluster, replace the mean by
            After averaging each cluster, the mean value computedcan be replaced

    """
    def __init__(self, k=5,
                 tol=0.001,
                 max_iter=300,
                 distance_function='euclidean',
                 averaging_function='mean',
                 centroid_replacement=False,
                 init_method='kmeans++',
                 plot_progress=False):
        self.distance_function = distance_function
        self.averaging_function = averaging_function
        self.centroid_replacement = centroid_replacement
        self.init_method = init_method
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.cost = 0
        self.labels = []
        self.centroids = {}
        # the distance of each datapoint to the closest cluster center
        self.distances = []
        # number of datapoints
        self.N = None
        self.plot_progress = plot_progress

    def cluster_init(self, data):
        """
        Several cluster initialization methods.
            'k-first': take the first k datapoints as centroids.
            'random': take k datapoints at random from the dataset.
            'kmeans++': See: "k-means++: The Advantages of Careful Seeding"    David Arthur ∗ Sergei Vassilvitskii†
        """
        if self.init_method == 'k-first':
            for i in range(self.k):
                self.centroids[i] = data[i]
        elif self.init_method == 'random':
            # generate uniform random indexes from 0 to N datapoints with size k
            indexes = np.random.randint(self.N, size=self.k)
            for i in range(self.k):
                self.centroids[i] = data[indexes[i]]
        elif self.init_method == 'kmeans++':
            self.kmeanspp_init(data)
        else:
            raise('UNKNOWN CLUSTER INITIALIZATION METHOD. Please use k-first, random or kmeans++')

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
        # init centroids according to selected method
        self.cluster_init(data)

        # perform max iterations of the algorithm
        for i in range(self.max_iter):
            if self.plot_progress:
                self.plot(data, 'Iteration number: ' + str(i))
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
        An iteration of the kmeans algorithm consists of a call to predict and a call to update centroids. The cost
        (inertia) is also computed on each iteration.
        :param data:
        :return:
        """
        [self.labels, self.distances] = self.predict(data)
        self.update_centroids(data)
        # compute the total cost of the current assigment based on the previous distances
        cost = np.sum(self.distances)
        # cost = np.average(self.distances)
        return cost

    def predict(self, data):
        """
        Try to speed the assignation of datapoints to clusters.
        In order to do this we compute the distance of each cluster to all the datapoints.
        Next, the min distance datapoint-cluster is used for the assignation.
        The assignation is based

        datapoint_distances returns the distance of each datapoint to the closest centroid

        :param data:
        :return:
        """
        distance_mat = []
        labels = -1*np.ones(self.N, dtype=np.int8)
        # find the distances of all datapoints to each cluster k
        # for each cluster, the operation computes the distance to all datapoints
        for k in range(self.k):
            centroid_k = self.centroids[k]
            # distances of all datapoints to centroid k
            d = self.compute_distance_function(data, centroid_k)
            distance_mat.append(d)
        distance_mat = np.array(distance_mat)
        datapoint_distances = np.zeros(self.N)
        # next, we select the min distance of each datapoint to each cluster
        for i in range(self.N):
            col = distance_mat[:, i]
            min_dist = np.amin(col)
            k_i = np.where(col == min_dist)
            labels[i] = k_i[0][0]
            datapoint_distances[i] = min_dist
        return labels, datapoint_distances

    def update_centroids(self, data):
        """
            find the position of the new centroids for the next iteration of the algorithm.

            - 'mean': In the vanilla k-means this is done by averaging all the datapoints assigned to each cluster.
            - 'mean-round': This an averaging following by a round operation. Makes sense for some special kind of data
                            that may be integer but still could be averaged and discretized to form the new cluster
                            centers.
            - 'binary-k-majority': This was specially developed for visual binary descriptors like ORB. The re
            desc1 = [179 252  33  50 113 253  75 238  59 123  32  29  63  22  14 181 179 139, 213  71 239  78 120 191 179 223
            24 186 120  85   7 253]

            desc2 = [ 52 185 252 112 139 205  85 223 175 189 172  41  30 245  78  22 171 146, 237  16  26  41 171  51 123 223
            21 127 227  66 198 227]

            convert to

                A k-majority works at the bit level

        :param data:
        :return:
        """
        if self.averaging_function == 'mean':
            for k in range(0, self.k):
                # find indexes that have been assigned to cluster i
                indexes = np.where(self.labels == k)
                # get data that corresponds to these indexes and compute an average, leave the results as is
                self.centroids[k] = np.average(data[indexes], axis=0)
        elif self.averaging_function == 'mean-round':
            for k in range(0, self.k):
                # find indexes that have been assigned to cluster i
                indexes = np.where(self.labels == k)
                # compute mean value
                centroid_k = np.average(data[indexes], axis=0)
                centroid_k = np.round(centroid_k)
                # round to closest value
                self.centroids[k] = centroid_k.astype(np.uint8)
        # if centroid replacement, find the closest existing sample of the dataset to replace the above computed
        # centroid. Use the member pre
        if self.centroid_replacement:
            # for each of the cluster centroids, find the closest example in the dataset and replace it
            # this makes sense for binary descriptors and avoids having descriptors that do not represent any
            # existing element in the dataset.
            for k in range(0, self.k):
                data_k = self.find_closest(data, self.centroids[k])
                self.centroids[k] = data_k

    def find_closest(self, data, centroid_k):
        """
        Find the closest sample of data to the centroid
        :param data:
        :return:
        """
        # distances of all datapoints to centroid k
        d = self.compute_distance_function(data, centroid_k)
        k_i = np.where(d == np.amin(d))
        index = k_i[0][0]
        return data[index, :]

    def compute_distance_function(self, data, centroid):
        """
        Compute distance between all data and a given centroid.
        In the euclidean distance, np.linalg.norm allows to compute the distance of all data to a centroid by computing
        a difference and then computing the L2-norm.
        In the case of the Hamming distance, the np.bitwise_xor has to be called for each descriptor independently, thus
        slowing down
        """
        if self.distance_function == 'euclidean':
            # returns an array of 1 x N, where N is the number of datapoints in data
            d = np.linalg.norm(data - centroid, axis=1)
            return d
        elif self.distance_function == 'hamming':
            # iterate along samples to compute the hamming distance
            d = self.compute_hamming_distances(data, centroid)
            return d

    def compute_hamming_distances(self, data, centroid):
        distances = -1 * np.ones(self.N, dtype=np.uint8)
        try:
            assert (data.dtype == np.uint8)
            assert (centroid.dtype == np.uint8)
        except AssertionError:
            print("Please use binary data (numpy.uint8) if Hamming distance is used.")
            print("Alternatively, use mean-round as averaging method is used.")
            print("Exiting")
            exit()
        # compute hamming distance for all descriptors in the dataset to the given centroid
        for i in range(self.N):
            sample = data[i]
            distances[i] = hamming(sample, centroid)
        return distances

    def kmeanspp_init(self, data):
        # store the desired number of clusters to be initialized
        K = self.k
        # select first cluster at random
        index = np.random.randint(self.N, size=1)
        self.centroids[0] = data[index[0]]
        for k in range(1, K):
            # the self.k must be set, since we are initializing and have not set the
            # total number of desired clusters.
            self.k = k
            self.centroids[k] = self.kmeanspp_select_new_centroid(data)
        self.k = K

    def kmeanspp_select_new_centroid(self, data):
        # obtain all distances to the closest centroids
        [labels, datapoint_distances] = self.predict(data)
        # square distances
        for i in range(self.N):
            datapoint_distances[i] = datapoint_distances[i]*datapoint_distances[i]
        # compute Sum D(x)2
        SDx2 = np.sum(datapoint_distances)
        probabilities = []
        for i in range(self.N):
            p = datapoint_distances[i]/SDx2
            probabilities.append(p)
        # now choice from the prior list with probability equal to p
        # no need to remove previous indexes, since, for each centroid the probability is zero
        sampleindex = np.random.choice(range(0, self.N), 1, p=probabilities)
        return data[sampleindex[0]]

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

    def plot(self, X, title):
        """
        Caution, plotting only features 1 and 2 of the dataset.
        Data X is predicted and then plotted
        :param X:
        :return:
        """
        [labels, _] = self.predict(X)
        plt.figure(1)
        # plot current centroids
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
                        marker="o", color="k", s=150, linewidths=5)

        colors = 10*['r', 'g', 'b', 'k', 'c', 'm', 'y']
        for k in range(0, self.k):
            cluster_data = (labels == k)
            col = colors[k]
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1],
                        c=col, marker='.', s=10)
        plt.title(title)
        plt.show()





