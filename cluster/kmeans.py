"""

A Vanilla KMeans Algorithm.

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
import time

class KMeans():
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

        # Init k centroids, may be done more efficiently if this were random
        for i in range(self.k):
            self.centroids[i] = data[i]

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
        :param data:
        :return:
        """
        # maintain a list where each datapoint points to one of the clusters.
        # labels are initialized to -1. Thus all -1 should be replaced with its corresponding classes.
        self.labels = -1*np.ones(self.N, dtype=np.int8)
        self.distances = np.zeros(self.N)
        # assign each datapoint to the closest centroid (assign each datapoint to a cluster)
        # each label[i] stores the cluster associated to the datapoint data[i]
        # [self.labels, self.distances] = self.predict(data)
        [self.labels, self.distances] = self.predict2(data)
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
            for centroid in self.centroids:
                distances_to_centroids.append(np.linalg.norm(datapoint - self.centroids[centroid]))
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





