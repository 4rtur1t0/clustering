"""
TODO:

"""
from itertools import cycle

import numpy as np
from cluster.kmeans import KMeans
import matplotlib.pyplot as plt


class Node():
    def __init__(self, id, n_clusters, kmeans_params):
        self.children = []
        self.parent = None
        # self.data = data
        self.indexes = None
        self.labels = None
        self.centroids = {}
        self.id = id
        self.node_type = 'node'
        self.n_clusters = n_clusters
        self.kmeans_params = kmeans_params
        self.cost = None
        self.kmeans = None

    def init_indexes(self, data):
        self.indexes = np.arange(0, len(data))

    def add_child(self, obj):
        """
        Each node has a list of children
        """
        self.children.append(obj)

    def add_parent(self, obj):
        """
        Each Node has a single parent.
        """
        self.parent = obj

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def get_n_datapoints(self):
        return len(self.indexes)

    def plot_data(self, data, color):
        # if no data is assigned to this node --> skip
        if self.indexes is None:
            return
        # plt.figure(1)
        X = data[self.indexes]
        for k in range(self.n_clusters):
            # col = colors[k]
            cluster_data = (self.labels == k)
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=color, marker='.', s=20)
            plt.scatter(self.centroids[k][0], self.centroids[k][1], c='k', marker='.', s=150)
        # plt.title("Data clustered at level" + str(level))
        # plt.show()

    def fit(self, data):
        # in this case, data has not been propagated to this node and no clustering needs to be performed
        # the node is not deleted
        if self.indexes is None:
            return
        # obtain the data that corresponds to the current node
        data = data[self.indexes]
        # check whether there are more clusters than data. In this case the clustering makes no sense and all datapoints
        # are selected as centroid with n_clusters = len(data)
        print('Clustering len(data)', len(data))
        if self.n_clusters > len(data):
            print('WARNING: cannot find more clusters than number of data.')
            print('WARNING: setting all datapoints as clusters.')
            self.n_clusters = len(data)
        km = KMeans(k=self.n_clusters, tol=self.kmeans_params.get('tol'), max_iter=self.kmeans_params.get('max_iter'),
                    distance_function=self.kmeans_params.get('distance_function'),
                    centroid_replacement=self.kmeans_params.get('centroid_replacement'),
                    init_method=self.kmeans_params.get('init_method'),
                    plot_progress=self.kmeans_params.get('plot_progress'))
        # # cluster the data
        # km = KMeans(k=self.n_clusters, tol=0.001, max_iter=500,
        #             distance_function='euclidean',
        #             centroid_replacement=False,
        #             init_method='kmeans++',
        #             # init_method='random',
        #             plot_progress=True)
        # fit current node with the data given
        km.fit(data)
        self.kmeans = km
        self.centroids = km.centroids
        self.labels = km.labels
        self.cost = km.cost

        if self.n_clusters == 1:
            print('debug')

        # if this is a leaf node, then finish, do not propagate indexes below
        if self.node_type == 'leaf':
            return

        # once clustered, propagate the data_indexes to this node's children
        # there are kw children of this node
        # caution, in the case where n_clusters is less than kw, some of the children nodes will not have data assigned
        # no subsequent clustering is performed for these Nodes
        for k in range(self.n_clusters):
            tf_indexes = (self.labels == k)
            self.children[k].indexes = self.indexes[tf_indexes]



class HKMeans():
    """
    Create a tree with branching factor kw with levels lw
    The tree stores the data
    """
    def __init__(self, kmeans_params, name='root', kw=2, lw=3):
        self.kw = kw
        self.Lw = lw
        self.name = name
        self.parent = None
        self.tree = {}
        self.centroids = None
        self.labels = None
        self.kmeans_params = kmeans_params

        self.create_tree()

    def cluster_level(self, nodes, data):
        """
        cluster each of the levels
        """
        # cluster each of the nodes at this tree level
        for node in nodes:
            node.fit(data)

    def fit(self, data):
        """
        cluster all levels
        """
        # init indexes at root. The root node possesses all indexes of the whole dataset
        root = self.tree[0][0]
        root.init_indexes(data)
        for level in range(self.Lw):
            print('Processing tree level: ', level, ' out of', self.Lw)
            nodes = self.tree[level]
            self.cluster_level(nodes, data)
        print('Finished!')
        self.compute_global_labels(data)
        self.save_centroids()

    def predict(self, data):
        """
        Returns the label of the closest centroid of the leaves.
        predict works in a top down fashion. Starts at the top node and works down.
        This reduces the number of comparisons that need to be performed.
        """
        node = self.tree[0][0]
        labels = -1*np.ones(len(data), dtype=int)
        distances = np.zeros(len(data))

        for i in range(len(data)):
            while True:
                print('Predict')
                # if node.node_type == 'node' or node.node_type == 'root':
                label, distance = node.kmeans.predict([data[i]])
                # switch to node below
                node = node.children[label[0]]
            global_label
            labels[i] = label
            distances[i] = distance[0]
        return labels, distances

    def create_tree(self):
        j = 1
        for level in range(0, self.Lw):
            nodes = []
            # create nodes and assign each node to the parent at the previous level
            for k in range(np.power(self.kw, level)):
                # create current node
                curr_node = Node(str(j), n_clusters=self.kw, kmeans_params=self.kmeans_params)
                # add parent and childrens to the nodes
                if level > 0:
                    parent_index = int(np.floor(k / self.kw))
                    # select parent from previous level
                    parent = self.tree[level - 1][parent_index]
                    # select parent
                    curr_node.add_parent(parent)
                    # add children
                    parent.add_child(curr_node)
                # the root node has no parent
                if level == 0:
                    curr_node.node_type = 'root'
                # tag as leaf the last nodes
                if level == self.Lw-1:
                    curr_node.node_type = 'leaf'
                # save to the list
                nodes.append(curr_node)
                j += 1
            self.tree[level] = nodes

    def print_tree(self):
        for level in range(self.Lw):
            nodes = self.tree[level]
            print(30*"_")
            for node in nodes:
                try:
                    print(node.id, '(', node.node_type, ')', end='')
                    print('(', node.get_n_datapoints(), ') ', end='')
                except TypeError:
                    continue
            print()
            print(30 * "_")

    def print_words(self):
        # print, now print the leaves (leaves are the centroids of the last kmeans nodes.
        print("WORDS: (centroids at the last depth level)")
        node_leaves = self.tree[self.Lw-1]
        i = 0
        for node_leaf in node_leaves:
            n_centroids = len(node_leaf.centroids)
            for k in range(n_centroids):
                print('WORD: ', i)
                print(node_leaf.centroids[k])
                i += 1

    def compute_global_labels(self, data):
        """
        Assign labels to every datapoint in data.
        We do not use the middle nodes for classification and stick to the leaf nodes.
        """
        self.labels = -1*np.ones(len(data), dtype=int)
        node_leaves = self.tree[self.Lw-1]
        word_i = 0
        for node_leaf in node_leaves:
            n_centroids = len(node_leaf.centroids)
            for k in range(n_centroids):
                idx_tf = node_leaf.labels == k
                idx = node_leaf.indexes[idx_tf]
                self.labels[idx] = int(word_i)
                # increment and switch to next word
                word_i += 1

    def save_centroids(self):
        node_leaves = self.tree[self.Lw-1]
        centroids = []
        for node_leaf in node_leaves:
            n_centroids = len(node_leaf.centroids)
            for k in range(n_centroids):
                centroids.append(node_leaf.centroids[k])
        self.centroids = np.array(centroids)

    def plot_tree_data(self, X):
        colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
        iterator = cycle(colors)
        for level in range(self.Lw):
            plt.figure(level)
            nodes = self.tree[level]
            for node in nodes:
                node.plot_data(X, color=next(iterator))
            plt.title('Tree at level ' + str(level))
            plt.show()

    def get_number_of_nodes(self):
        n_nodes = 0
        for level in range(self.Lw):
            n_nodes += len(self.tree[level])
        return n_nodes

    def get_number_of_leaves(self):
        n_leaves = len(self.tree[self.Lw-1])
        return n_leaves

    def get_expected_number_of_words(self):
        n_words = np.power(self.kw, self.Lw)
        return n_words

    def count_number_of_words(self):
        leaves = self.tree[self.Lw - 1]
        n_words = 0
        for leaf in leaves:
            n_words += len(leaf.centroids)
        return n_words

    def get_total_cost(self):
        leaves = self.tree[self.Lw - 1]
        cost = 0
        for leaf in leaves:
            if leaf.cost is None:
                continue
            cost += leaf.cost
        return cost

    def get_n_datapoints_per_leaf_distribution(self):
        """
        Gets the number of datapoints at each leaf. These datapoints are further clustered in kw clusters according to
        the result in labels.
        """
        node_leaves = self.tree[self.Lw-1]
        datapoints = []
        for node_leaf in node_leaves:
            try:
                n = node_leaf.get_n_datapoints()
                datapoints.append(n)
            except:
                pass
                # print('Found node_leaf without data, continuing')
        datapoints = np.array(datapoints)
        return np.mean(datapoints), np.cov(datapoints)

    def get_n_datapoints_per_word_distribution(self):
        """
        Get the number of points associated to each of the clusters at the lowest level.
        """
        node_leaves = self.tree[self.Lw-1]
        datapoints = []
        for node_leaf in node_leaves:
            for k in set(node_leaf.labels):
                idx_tf = node_leaf.labels == k
                n_datapoints = len(node_leaf.indexes[idx_tf])
                datapoints.append(n_datapoints)
        datapoints = np.array(datapoints)
        return np.mean(datapoints), np.cov(datapoints)







